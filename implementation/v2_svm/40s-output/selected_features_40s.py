#!/usr/bin/env python3
"""
Train an SVM on selected 40-second HRV features with a lighter hyperparameter
grid using Leave-One-Subject-Out validation.
"""

import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class SelectedFeature40SSVMTrainer:
    def __init__(self, data_directory=None):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or os.path.normpath(
            os.path.join(self.script_directory, "..", "40s-output")
        )
        self.selected_features = [
            "sd1_sd2_ratio",
            "lf_power",
            "nn_min",
            "dfa",
            "sd2",
        ]

    def parse_filename(self, filename):
        pattern = (
            r"pr_([^_]+)_([^_]+)_(truth|lie)-sequence_"
            r"combined_hrv_v2_svm_40s\.csv"
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
            if filename.endswith("_combined_hrv_v2_svm_40s.csv")
        )

        file_metadata = []
        for filename in csv_files:
            metadata = self.parse_filename(filename)
            if metadata:
                file_metadata.append(metadata)

        if not file_metadata:
            raise ValueError(f"No 40-second HRV CSV files found in {self.data_directory}")

        print(f"Discovered {len(file_metadata)} 40-second HRV CSV files")
        print("Subjects found:", sorted({metadata["subject"] for metadata in file_metadata}))
        print("Conditions found:", sorted({metadata["condition"] for metadata in file_metadata}))
        return file_metadata

    def load_data(self, file_metadata):
        all_dataframes = []

        for metadata in file_metadata:
            filepath = os.path.join(self.data_directory, metadata["filename"])
            dataframe = pd.read_csv(filepath)
            dataframe["subject"] = metadata["subject"]
            dataframe["condition"] = metadata["condition"]
            dataframe["label"] = metadata["label"]
            dataframe["binary_label"] = 1 if metadata["label"] == "lie" else 0
            dataframe["source_file"] = metadata["filename"]
            all_dataframes.append(dataframe)

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
        return feature_matrix, labels, groups, imputer

    def optimize_and_train(self, feature_matrix, labels, groups):
        print("\n" + "=" * 60)
        print("LIGHTWEIGHT SVM HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)

        logo = LeaveOneGroupOut()
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

        param_grid = [
            {
                "svm__kernel": ["linear"],
                "svm__C": [0.1, 1, 10],
            },
            {
                "svm__kernel": ["rbf"],
                "svm__C": [0.1, 1, 10, 100],
                "svm__gamma": ["scale", 0.01, 0.1, 1],
            },
        ]

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=logo,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            verbose=1,
        )
        grid_search.fit(feature_matrix, labels, groups=groups)

        best_estimator = grid_search.best_estimator_
        best_params = {
            key.replace("svm__", ""): value
            for key, value in grid_search.best_params_.items()
        }

        y_true_all = []
        y_pred_all = []
        decision_scores_all = []
        fold_accuracies = []
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
            decision_scores = model.decision_function(x_test)
            fold_accuracy = accuracy_score(y_test, y_pred)

            subject_results[test_subject] = {
                "accuracy": fold_accuracy,
                "n_samples": int(len(y_test)),
            }
            fold_accuracies.append(fold_accuracy)
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
            decision_scores_all.extend(np.asarray(decision_scores).tolist())

            print(
                f"Subject {test_subject}: accuracy={fold_accuracy:.4f} "
                f"({len(y_test)} samples)"
            )

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        overall_f1 = f1_score(y_true_all, y_pred_all)
        overall_auc = roc_auc_score(y_true_all, decision_scores_all)
        losocv_std = float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0
        overall_confusion_matrix = confusion_matrix(y_true_all, y_pred_all)

        return {
            "overall_accuracy": overall_accuracy,
            "best_cv_score": grid_search.best_score_,
            "best_params": best_params,
            "selected_features": self.selected_features,
            "subject_results": subject_results,
            "fold_accuracies": fold_accuracies,
            "losocv_std": losocv_std,
            "auc": overall_auc,
            "f1_score": overall_f1,
            "confusion_matrix": overall_confusion_matrix,
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
            "selected_feature_train_40s_results.csv",
        )
        pd.DataFrame(
            [
                {
                    "loso_accuracy": results["overall_accuracy"],
                    "losocv_standard_deviation": results["losocv_std"],
                    "auc": results["auc"],
                    "f1_score": results["f1_score"],
                    "best_cv_score": results["best_cv_score"],
                    "selected_features": ", ".join(results["selected_features"]),
                    "best_params": str(results["best_params"]),
                    "confusion_matrix": np.array2string(results["confusion_matrix"]),
                }
            ]
        ).to_csv(results_path, index=False)

        subject_path = os.path.join(
            self.script_directory,
            "selected_feature_train_40s_subject_results.csv",
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
            "selected_feature_train_40s_grid_search_results.csv",
        )
        results["cv_results"].sort_values("rank_test_score").to_csv(grid_path, index=False)

        model_path = os.path.join(
            self.script_directory,
            "selected_feature_train_40s_model.joblib",
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
        print("=== SELECTED FEATURE SVM TRAINING (40S) ===")
        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)
        feature_matrix, labels, groups, imputer = self.prepare_features(dataframe)
        results = self.optimize_and_train(feature_matrix, labels, groups)

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Best CV Accuracy: {results['best_cv_score']:.4f}")
        print(f"Final LOSO Accuracy: {results['overall_accuracy']:.4f}")
        print(f"LOSOCV Standard Deviation: {results['losocv_std']:.4f}")
        print(f"AUC: {results['auc']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
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
    trainer = SelectedFeature40SSVMTrainer()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()