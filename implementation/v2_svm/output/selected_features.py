#!/usr/bin/env python3
"""
Train an SVM only on statistically selected HRV features with aggressive
hyperparameter optimization using Leave-One-Subject-Out validation.
"""

import os
import re
import warnings
from datetime import datetime

from joblib import parallel_backend
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class SelectedFeatureSVMTrainer:
    def __init__(self, data_directory=None):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or self.script_directory
        self.cpu_jobs = os.cpu_count() or 1
        self.progress_log_path = os.path.join(
            self.script_directory,
            "selected_feature_train_progress.log",
        )
        self.selected_features = [
            "sd1_sd2_ratio",
            "lf_power",
            "nn_min",
            "dfa",
            "sd2",
        ]
        self.backend_name = "cpu-sklearn"
        self.svm_class = SVC

    def log_progress(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        with open(self.progress_log_path, "a", encoding="ascii", errors="ignore") as log_file:
            log_file.write(formatted_message + "\n")

    def initialize_backend(self, prefer_gpu=True):
        self.backend_name = "cpu-sklearn"
        self.svm_class = SVC

        if not prefer_gpu:
            self.log_progress("GPU mode disabled; using sklearn CPU backend")
            return

        try:
            from thundersvm import SVC as ThunderSVC

            self.svm_class = ThunderSVC
            self.backend_name = "gpu-thundersvm"
            self.log_progress("Using ThunderSVM GPU backend")
            return
        except ImportError:
            self.log_progress(
                "ThunderSVM is not installed; falling back to sklearn CPU backend"
            )
        except Exception as error:
            self.log_progress(
                f"ThunderSVM is installed but unusable ({error}); falling back to sklearn CPU backend"
            )

    def create_svm_model(self, svm_params):
        base_params = dict(svm_params)
        base_params.setdefault("kernel", "rbf")

        if self.backend_name == "gpu-thundersvm":
            base_params.setdefault("cache_size", 1024)
            return make_pipeline(StandardScaler(), self.svm_class(**base_params))

        base_params.setdefault("class_weight", "balanced")
        base_params.setdefault("random_state", 42)
        base_params.setdefault("cache_size", 1024)
        return make_pipeline(StandardScaler(), self.svm_class(**base_params))

    def parse_filename(self, filename):
        pattern = (
            r"pr_([^_]+)_([^_]+)_(truth|lie)-sequence_"
            r"combined_hrv_v2_svm\.csv"
        )
        match = re.match(pattern, filename)
        if not match:
            return None

        return {
            "subject": match.group(1),
            "condition": match.group(2),
            "label": match.group(3),
            "filename": filename,
        }

    def discover_files(self):
        csv_files = sorted(
            filename
            for filename in os.listdir(self.data_directory)
            if filename.endswith("_combined_hrv_v2_svm.csv")
        )

        file_metadata = []
        for filename in csv_files:
            metadata = self.parse_filename(filename)
            if metadata:
                file_metadata.append(metadata)

        if not file_metadata:
            raise ValueError(
                f"No v2 combined HRV CSV files found in {self.data_directory}"
            )

        print(f"Discovered {len(file_metadata)} v2 combined HRV CSV files")
        print(
            "Subjects found:",
            sorted({metadata["subject"] for metadata in file_metadata}),
        )
        print(
            "Conditions found:",
            sorted({metadata["condition"] for metadata in file_metadata}),
        )
        return file_metadata

    def load_data(self, file_metadata):
        all_dataframes = []

        for metadata in file_metadata:
            filepath = os.path.join(self.data_directory, metadata["filename"])

            try:
                dataframe = pd.read_csv(filepath)
            except Exception as error:
                print(f"Error loading {metadata['filename']}: {error}")
                continue

            dataframe["subject"] = metadata["subject"]
            dataframe["condition"] = metadata["condition"]
            dataframe["label"] = metadata["label"]
            dataframe["binary_label"] = 1 if metadata["label"] == "lie" else 0
            dataframe["source_file"] = metadata["filename"]
            all_dataframes.append(dataframe)

        if not all_dataframes:
            raise ValueError("No valid CSV files could be loaded")

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} total samples")
        print(
            "Label distribution:",
            combined_df["binary_label"].value_counts().sort_index().to_dict(),
        )
        return combined_df

    def prepare_features(self, dataframe):
        missing_features = [
            feature for feature in self.selected_features if feature not in dataframe.columns
        ]
        if missing_features:
            raise ValueError(f"Missing selected features: {missing_features}")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(dataframe[self.selected_features])
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()

        print(f"Using selected features: {self.selected_features}")
        print(
            f"Final dataset: {feature_matrix.shape[0]} samples, "
            f"{feature_matrix.shape[1]} features"
        )
        print(
            f"Label distribution: Truth={np.sum(labels == 0)}, "
            f"Lie={np.sum(labels == 1)}"
        )

        return feature_matrix, labels, groups, imputer

    def optimize_model_cpu(self, feature_matrix, labels, groups, param_grid):
        self.log_progress(f"Using {self.cpu_jobs} CPU workers for threaded grid search")

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        class_weight="balanced",
                        random_state=42,
                        cache_size=1024,
                    ),
                ),
            ]
        )

        prefixed_param_grid = []
        for grid in param_grid:
            prefixed_param_grid.append({f"svm__{key}": value for key, value in grid.items()})

        logo = LeaveOneGroupOut()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=prefixed_param_grid,
            cv=logo,
            scoring="accuracy",
            n_jobs=self.cpu_jobs,
            refit=True,
            return_train_score=False,
            verbose=2,
            pre_dispatch=self.cpu_jobs,
        )
        with parallel_backend("threading", n_jobs=self.cpu_jobs):
            grid_search.fit(feature_matrix, labels, groups=groups)

        best_estimator = grid_search.best_estimator_
        best_params = {
            key.replace("svm__", ""): value
            for key, value in grid_search.best_params_.items()
        }
        cv_results = pd.DataFrame(grid_search.cv_results_)
        return best_estimator, best_params, grid_search.best_score_, cv_results

    def optimize_model_gpu(self, feature_matrix, labels, groups, param_grid):
        logo = LeaveOneGroupOut()
        parameter_sets = list(ParameterGrid(param_grid))
        total_candidates = len(parameter_sets)
        total_folds = logo.get_n_splits(groups=groups)

        self.log_progress(
            f"GPU manual search starting with {total_candidates} candidates and {total_folds} LOSO folds"
        )

        cv_rows = []
        best_score = -np.inf
        best_params = None

        for candidate_index, svm_params in enumerate(parameter_sets, start=1):
            self.log_progress(
                f"Candidate {candidate_index}/{total_candidates}: {svm_params}"
            )
            fold_scores = []

            for fold_index, (train_index, test_index) in enumerate(
                logo.split(feature_matrix, labels, groups),
                start=1,
            ):
                model = self.create_svm_model(svm_params)
                x_train = feature_matrix[train_index]
                x_test = feature_matrix[test_index]
                y_train = labels[train_index]
                y_test = labels[test_index]

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                score = accuracy_score(y_test, y_pred)
                fold_scores.append(score)

                self.log_progress(
                    f"Candidate {candidate_index}/{total_candidates}, fold {fold_index}/{total_folds}: accuracy={score:.4f}"
                )

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            row = {f"param_svm__{key}": value for key, value in svm_params.items()}
            row["mean_test_score"] = mean_score
            row["std_test_score"] = std_score
            cv_rows.append(row)

            if mean_score > best_score:
                best_score = mean_score
                best_params = dict(svm_params)
                self.log_progress(
                    f"New best candidate: score={best_score:.4f}, params={best_params}"
                )

        cv_results = pd.DataFrame(cv_rows).sort_values(
            by="mean_test_score",
            ascending=False,
        )
        cv_results["rank_test_score"] = range(1, len(cv_results) + 1)
        best_estimator = self.create_svm_model(best_params)
        best_estimator.fit(feature_matrix, labels)
        return best_estimator, best_params, best_score, cv_results

    def optimize_and_train(self, feature_matrix, labels, groups):
        print(f"\n{'=' * 60}")
        print("AGGRESSIVE SVM HYPERPARAMETER OPTIMIZATION")
        print(f"{'=' * 60}")

        if os.path.exists(self.progress_log_path):
            os.remove(self.progress_log_path)
        self.log_progress(f"Backend selected: {self.backend_name}")

        param_grid = [
            {
                "kernel": ["linear"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            },
            {
                "kernel": ["rbf"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "gamma": ["scale", "auto", 1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
            {
                "kernel": ["poly"],
                "C": [0.01, 0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1],
                "degree": [2, 3, 4, 5],
                "coef0": [0.0, 0.5, 1.0, 2.0],
            },
            {
                "kernel": ["sigmoid"],
                "C": [0.01, 0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1],
                "coef0": [0.0, 0.5, 1.0, 2.0],
            },
        ]

        if self.backend_name == "gpu-thundersvm":
            best_estimator, best_params, best_cv_score, cv_results = self.optimize_model_gpu(
                feature_matrix,
                labels,
                groups,
                param_grid,
            )
        else:
            self.log_progress(
                "CUDA note: sklearn SVC uses CPU only, so GPU is not used in this environment"
            )
            best_estimator, best_params, best_cv_score, cv_results = self.optimize_model_cpu(
                feature_matrix,
                labels,
                groups,
                param_grid,
            )

        print(f"Best cross-validated accuracy: {best_cv_score:.4f}")
        print(f"Best parameters: {best_params}")

        y_true_all = []
        y_pred_all = []
        subject_results = {}

        logo = LeaveOneGroupOut()

        for train_index, test_index in logo.split(feature_matrix, labels, groups):
            model = clone(best_estimator)

            x_train = feature_matrix[train_index]
            x_test = feature_matrix[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]
            test_subject = groups[test_index][0]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            subject_results[test_subject] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "n_samples": int(len(y_test)),
            }
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

            print(
                f"Subject {test_subject}: "
                f"{subject_results[test_subject]['accuracy']:.4f} "
                f"({subject_results[test_subject]['n_samples']} samples)"
            )

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)

        return {
            "overall_accuracy": overall_accuracy,
            "best_cv_score": best_cv_score,
            "best_params": best_params,
            "selected_features": self.selected_features,
            "subject_results": subject_results,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "cv_results": cv_results,
            "final_model": best_estimator,
        }

    def save_results(self, results, imputer):
        results_path = os.path.join(
            self.script_directory,
            "selected_feature_train_results.csv",
        )
        pd.DataFrame(
            [
                {
                    "loso_accuracy": results["overall_accuracy"],
                    "best_cv_score": results["best_cv_score"],
                    "selected_features": ", ".join(results["selected_features"]),
                    "best_params": str(results["best_params"]),
                }
            ]
        ).to_csv(results_path, index=False)

        subject_path = os.path.join(
            self.script_directory,
            "selected_feature_train_subject_results.csv",
        )
        pd.DataFrame(
            [
                {
                    "subject": subject,
                    "accuracy": result["accuracy"],
                    "n_samples": result["n_samples"],
                }
                for subject, result in sorted(results["subject_results"].items())
            ]
        ).to_csv(subject_path, index=False)

        grid_path = os.path.join(
            self.script_directory,
            "selected_feature_train_grid_search_results.csv",
        )
        results["cv_results"].sort_values("rank_test_score").to_csv(grid_path, index=False)

        model_path = os.path.join(
            self.script_directory,
            "selected_feature_train_model.joblib",
        )
        artifact = {
            "model": results["final_model"],
            "imputer": imputer,
            "selected_features": results["selected_features"],
            "best_params": results["best_params"],
        }
        joblib.dump(artifact, model_path)

        print("\nSaved outputs:")
        print(f"  - {results_path}")
        print(f"  - {subject_path}")
        print(f"  - {grid_path}")
        print(f"  - {model_path}")

    def run(self):
        print("=== SELECTED FEATURE SVM TRAINING ===")
        self.initialize_backend(prefer_gpu=True)
        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)
        feature_matrix, labels, groups, imputer = self.prepare_features(dataframe)
        results = self.optimize_and_train(feature_matrix, labels, groups)

        print(f"\n{'=' * 80}")
        print("FINAL RESULTS")
        print(f"{'=' * 80}")
        print(f"Best CV Accuracy: {results['best_cv_score']:.4f}")
        print(f"Final LOSO Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Selected Features: {results['selected_features']}")
        print("Best Parameters:")
        for key, value in results["best_params"].items():
            print(f"  - {key}: {value}")
        print("\nConfusion Matrix:")
        print(results["confusion_matrix"])
        print("\nClassification Report:")
        print(results["classification_report"])

        self.save_results(results, imputer)


def main():
    trainer = SelectedFeatureSVMTrainer()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()