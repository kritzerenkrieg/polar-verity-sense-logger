#!/usr/bin/env python3
"""
Train an SVM with aggressive feature selection and a one-candidate
hyperparameter grid using Leave-One-Subject-Out validation.
"""

import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class GridFeatureSVMTrainer:
    def __init__(self, data_directory=None):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or self.script_directory
        self.significance_table_path = os.path.join(
            self.script_directory,
            "feature_significance_table.csv",
        )
        self.feature_analysis_path = os.path.join(
            self.script_directory,
            "feature_importance_analysis.csv",
        )
        self.min_significance = "**"
        self.max_features = 3
        self.fixed_params = {
            "kernel": "rbf",
            "C": 1000,
            "gamma": 0.1,
        }

    @staticmethod
    def parse_filename(filename):
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

    @staticmethod
    def significance_rank(significance_marker):
        return {
            "": 0,
            "*": 1,
            "**": 2,
            "***": 3,
        }.get(str(significance_marker).strip(), 0)

    def select_features(self):
        if not os.path.exists(self.significance_table_path):
            raise FileNotFoundError(
                f"Missing significance table: {self.significance_table_path}"
            )
        if not os.path.exists(self.feature_analysis_path):
            raise FileNotFoundError(
                f"Missing feature analysis file: {self.feature_analysis_path}"
            )

        significance_df = pd.read_csv(self.significance_table_path)
        analysis_df = pd.read_csv(self.feature_analysis_path)

        significance_df["significance_rank"] = significance_df["significance"].map(
            self.significance_rank
        )
        analysis_columns = ["feature", "consensus_rank", "consensus_score"]
        ranked_df = significance_df.merge(
            analysis_df[analysis_columns],
            on="feature",
            how="left",
        )

        minimum_rank = self.significance_rank(self.min_significance)
        filtered_df = ranked_df[
            ranked_df["significance_rank"] >= minimum_rank
        ].copy()

        if filtered_df.empty:
            raise ValueError(
                f"No features met the minimum significance threshold {self.min_significance}"
            )

        filtered_df = filtered_df.sort_values(
            by=["significance_rank", "consensus_rank", "consensus_score"],
            ascending=[False, True, False],
            na_position="last",
        )
        selected_features = filtered_df["feature"].head(self.max_features).tolist()

        print("Aggressive feature selection summary:")
        print(f"  Minimum significance: {self.min_significance}")
        print(f"  Max features kept: {self.max_features}")
        print(f"  Selected features: {selected_features}")
        return selected_features

    def prepare_features(self, dataframe, selected_features):
        missing_features = [
            feature for feature in selected_features if feature not in dataframe.columns
        ]
        if missing_features:
            raise ValueError(f"Missing selected features: {missing_features}")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(dataframe[selected_features])
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()

        print(f"Using selected features: {selected_features}")
        print(
            f"Final dataset: {feature_matrix.shape[0]} samples, "
            f"{feature_matrix.shape[1]} features"
        )
        print(
            f"Label distribution: Truth={np.sum(labels == 0)}, "
            f"Lie={np.sum(labels == 1)}"
        )
        return feature_matrix, labels, groups, imputer

    def build_pipeline(self):
        return Pipeline(
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

    def optimize_and_train(self, feature_matrix, labels, groups, selected_features):
        print(f"\n{'=' * 60}")
        print("MINIMAL GRID SEARCH + AGGRESSIVE FEATURES")
        print(f"{'=' * 60}")

        param_grid = [
            {
                "svm__kernel": [self.fixed_params["kernel"]],
                "svm__C": [self.fixed_params["C"]],
                "svm__gamma": [self.fixed_params["gamma"]],
            }
        ]

        print("Grid search combinations: 1")
        print(f"Fixed hyperparameters: {self.fixed_params}")

        logo = LeaveOneGroupOut()
        pipeline = self.build_pipeline()
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=logo,
            scoring="accuracy",
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        grid_search.fit(feature_matrix, labels, groups=groups)

        best_estimator = grid_search.best_estimator_
        best_params = {
            key.replace("svm__", ""): value
            for key, value in grid_search.best_params_.items()
        }

        print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {best_params}")

        y_true_all = []
        y_pred_all = []
        subject_results = {}

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
            "best_cv_score": grid_search.best_score_,
            "best_params": best_params,
            "selected_features": selected_features,
            "subject_results": subject_results,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "cv_results": pd.DataFrame(grid_search.cv_results_),
            "final_model": best_estimator,
        }

    def save_results(self, results, imputer):
        results_path = os.path.join(
            self.script_directory,
            "grid_feature_train_results.csv",
        )
        pd.DataFrame(
            [
                {
                    "loso_accuracy": results["overall_accuracy"],
                    "best_cv_score": results["best_cv_score"],
                    "n_selected_features": len(results["selected_features"]),
                    "selected_features": ", ".join(results["selected_features"]),
                    "best_params": str(results["best_params"]),
                }
            ]
        ).to_csv(results_path, index=False)

        subject_path = os.path.join(
            self.script_directory,
            "grid_feature_train_subject_results.csv",
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
            "grid_feature_train_grid_search_results.csv",
        )
        results["cv_results"].sort_values("rank_test_score").to_csv(grid_path, index=False)

        model_path = os.path.join(
            self.script_directory,
            "grid_feature_train_model.joblib",
        )
        artifact = {
            "model": results["final_model"],
            "imputer": imputer,
            "selected_features": results["selected_features"],
            "best_params": results["best_params"],
            "selection_rule": {
                "min_significance": self.min_significance,
                "max_features": self.max_features,
            },
        }
        joblib.dump(artifact, model_path)

        print("\nSaved outputs:")
        print(f"  - {results_path}")
        print(f"  - {subject_path}")
        print(f"  - {grid_path}")
        print(f"  - {model_path}")

    def run(self):
        print("=== GRID FEATURE SVM TRAINING ===")
        selected_features = self.select_features()
        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)
        feature_matrix, labels, groups, imputer = self.prepare_features(
            dataframe,
            selected_features,
        )
        results = self.optimize_and_train(
            feature_matrix,
            labels,
            groups,
            selected_features,
        )

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
    trainer = GridFeatureSVMTrainer()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()