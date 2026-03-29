#!/usr/bin/env python3
"""
Train an SVM with fixed hyperparameters while aggressively searching
feature subset sizes from 22 features down to 1 using LOSO validation.
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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class FeatureTrainSVMTrainer:
    def __init__(self, data_directory=None):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or self.script_directory
        self.feature_analysis_path = os.path.join(
            self.script_directory,
            "feature_importance_analysis.csv",
        )
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

    def load_ranked_features(self):
        if not os.path.exists(self.feature_analysis_path):
            raise FileNotFoundError(
                f"Missing feature analysis file: {self.feature_analysis_path}"
            )

        analysis_df = pd.read_csv(self.feature_analysis_path)
        required_columns = {"feature", "consensus_rank", "consensus_score"}
        missing_columns = required_columns.difference(analysis_df.columns)
        if missing_columns:
            raise ValueError(
                f"Feature analysis is missing columns: {sorted(missing_columns)}"
            )

        ranked_df = analysis_df.sort_values(
            by=["consensus_rank", "consensus_score"],
            ascending=[True, False],
        )
        ranked_features = ranked_df["feature"].tolist()

        print("Consensus-ranked features:")
        for rank, feature_name in enumerate(ranked_features, start=1):
            print(f"  {rank:2d}. {feature_name}")

        return ranked_features

    def prepare_feature_matrix(self, dataframe, ranked_features):
        missing_features = [
            feature for feature in ranked_features if feature not in dataframe.columns
        ]
        if missing_features:
            raise ValueError(f"Missing ranked features in dataset: {missing_features}")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(dataframe[ranked_features])
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()

        print(
            f"Final dataset: {feature_matrix.shape[0]} samples, "
            f"{feature_matrix.shape[1]} ranked features"
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
                        kernel=self.fixed_params["kernel"],
                        C=self.fixed_params["C"],
                        gamma=self.fixed_params["gamma"],
                        class_weight="balanced",
                        random_state=42,
                        cache_size=1024,
                    ),
                ),
            ]
        )

    def evaluate_feature_count(self, feature_matrix, labels, groups, ranked_features, feature_count):
        selected_features = ranked_features[:feature_count]
        selected_matrix = feature_matrix[:, :feature_count]
        logo = LeaveOneGroupOut()
        pipeline = self.build_pipeline()

        fold_scores = []
        y_true_all = []
        y_pred_all = []
        subject_results = {}

        for train_index, test_index in logo.split(selected_matrix, labels, groups):
            model = clone(pipeline)
            x_train = selected_matrix[train_index]
            x_test = selected_matrix[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]
            test_subject = groups[test_index][0]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = accuracy_score(y_test, y_pred)

            fold_scores.append(score)
            subject_results[test_subject] = {
                "accuracy": score,
                "n_samples": int(len(y_test)),
            }
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        return {
            "feature_count": feature_count,
            "selected_features": selected_features,
            "mean_test_score": float(np.mean(fold_scores)),
            "std_test_score": float(np.std(fold_scores)),
            "overall_accuracy": overall_accuracy,
            "subject_results": subject_results,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "final_model": clone(pipeline).fit(selected_matrix, labels),
        }

    def optimize_and_train(self, feature_matrix, labels, groups, ranked_features):
        print(f"\n{'=' * 60}")
        print("AGGRESSIVE FEATURE COUNT SEARCH")
        print(f"{'=' * 60}")
        print(f"Fixed hyperparameters: {self.fixed_params}")
        print(f"Feature count candidates: {len(ranked_features)} down to 1")

        all_results = []
        best_result = None

        for feature_count in range(len(ranked_features), 0, -1):
            result = self.evaluate_feature_count(
                feature_matrix,
                labels,
                groups,
                ranked_features,
                feature_count,
            )
            all_results.append(
                {
                    "feature_count": result["feature_count"],
                    "mean_test_score": result["mean_test_score"],
                    "std_test_score": result["std_test_score"],
                    "overall_accuracy": result["overall_accuracy"],
                    "selected_features": ", ".join(result["selected_features"]),
                    "kernel": self.fixed_params["kernel"],
                    "C": self.fixed_params["C"],
                    "gamma": self.fixed_params["gamma"],
                }
            )

            print(
                f"Top-{feature_count:2d} features: "
                f"CV={result['mean_test_score']:.4f}, "
                f"LOSO={result['overall_accuracy']:.4f}"
            )

            if best_result is None:
                best_result = result
                continue

            if result["mean_test_score"] > best_result["mean_test_score"]:
                best_result = result
            elif result["mean_test_score"] == best_result["mean_test_score"]:
                if result["feature_count"] < best_result["feature_count"]:
                    best_result = result

        cv_results = pd.DataFrame(all_results).sort_values(
            by=["mean_test_score", "feature_count"],
            ascending=[False, True],
        )

        print(f"Best cross-validated accuracy: {best_result['mean_test_score']:.4f}")
        print(f"Best feature count: {best_result['feature_count']}")
        print(f"Best selected features: {best_result['selected_features']}")

        return {
            "overall_accuracy": best_result["overall_accuracy"],
            "best_cv_score": best_result["mean_test_score"],
            "best_params": dict(self.fixed_params),
            "selected_features": best_result["selected_features"],
            "subject_results": best_result["subject_results"],
            "confusion_matrix": best_result["confusion_matrix"],
            "classification_report": best_result["classification_report"],
            "cv_results": cv_results,
            "final_model": best_result["final_model"],
            "best_feature_count": best_result["feature_count"],
        }

    def save_results(self, results, imputer):
        results_path = os.path.join(
            self.script_directory,
            "feature_train_results.csv",
        )
        pd.DataFrame(
            [
                {
                    "loso_accuracy": results["overall_accuracy"],
                    "best_cv_score": results["best_cv_score"],
                    "best_feature_count": results["best_feature_count"],
                    "selected_features": ", ".join(results["selected_features"]),
                    "best_params": str(results["best_params"]),
                }
            ]
        ).to_csv(results_path, index=False)

        subject_path = os.path.join(
            self.script_directory,
            "feature_train_subject_results.csv",
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
            "feature_train_grid_search_results.csv",
        )
        results["cv_results"].to_csv(grid_path, index=False)

        model_path = os.path.join(
            self.script_directory,
            "feature_train_model.joblib",
        )
        artifact = {
            "model": results["final_model"],
            "imputer": imputer,
            "selected_features": results["selected_features"],
            "best_params": results["best_params"],
            "best_feature_count": results["best_feature_count"],
        }
        joblib.dump(artifact, model_path)

        print("\nSaved outputs:")
        print(f"  - {results_path}")
        print(f"  - {subject_path}")
        print(f"  - {grid_path}")
        print(f"  - {model_path}")

    def run(self):
        print("=== FEATURE TRAIN SVM TRAINING ===")
        ranked_features = self.load_ranked_features()
        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)
        feature_matrix, labels, groups, imputer = self.prepare_feature_matrix(
            dataframe,
            ranked_features,
        )
        results = self.optimize_and_train(
            feature_matrix,
            labels,
            groups,
            ranked_features,
        )

        print(f"\n{'=' * 80}")
        print("FINAL RESULTS")
        print(f"{'=' * 80}")
        print(f"Best CV Accuracy: {results['best_cv_score']:.4f}")
        print(f"Final LOSO Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Best Feature Count: {results['best_feature_count']}")
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
    trainer = FeatureTrainSVMTrainer()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()