#!/usr/bin/env python3
"""
Evaluate the legacy high-performing 12-feature HRV combination on the newer
40-second combined dataset, augmented with non-linear HRV features and a wider
SVM kernel search.
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


class SVMTrainNew:
    def __init__(self, data_directory=None):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or self.script_directory
        self.legacy_best_12 = [
            "nn_count",
            "nn_mean",
            "nn_min",
            "sdnn",
            "sdsd",
            "pnn20",
            "lf_power",
            "hf_power",
            "lf_hf_ratio",
            "lf_norm",
            "hf_norm",
            "lf_peak",
        ]
        self.nonlinear_features = ["sd1", "sd2", "sd1_sd2_ratio", "dfa"]
        self.combinations = self.build_feature_combinations()

    def build_feature_combinations(self):
        combinations = [
            {
                "name": "legacy_12feat",
                "features": self.legacy_best_12,
            }
        ]

        for feature in self.nonlinear_features:
            combinations.append(
                {
                    "name": f"legacy_12feat_plus_{feature}",
                    "features": self.legacy_best_12 + [feature],
                }
            )

        combinations.append(
            {
                "name": "legacy_12feat_plus_all_nonlinear",
                "features": self.legacy_best_12 + self.nonlinear_features,
            }
        )
        return combinations

    @staticmethod
    def parse_filename(filename):
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

    def prepare_matrix(self, dataframe, feature_list):
        missing_features = [feature for feature in feature_list if feature not in dataframe.columns]
        if missing_features:
            raise ValueError(f"Missing expected HRV features: {missing_features}")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(dataframe[feature_list])
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()
        return feature_matrix, labels, groups, imputer

    @staticmethod
    def build_param_grid():
        return [
            {
                "svm__kernel": ["linear"],
                "svm__C": [0.01, 0.1, 1, 10, 100],
            },
            {
                "svm__kernel": ["rbf"],
                "svm__C": [0.1, 1, 10, 100],
                "svm__gamma": ["scale", 0.01, 0.1, 1],
            },
            {
                "svm__kernel": ["poly"],
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", 0.01, 0.1],
                "svm__degree": [2, 3],
                "svm__coef0": [0.0, 1.0],
            },
            {
                "svm__kernel": ["sigmoid"],
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", 0.01, 0.1],
                "svm__coef0": [0.0, 1.0],
            },
        ]

    @staticmethod
    def build_pipeline():
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

    def evaluate_combination(self, feature_matrix, labels, groups, feature_names):
        logo = LeaveOneGroupOut()
        grid_search = GridSearchCV(
            estimator=self.build_pipeline(),
            param_grid=self.build_param_grid(),
            cv=logo,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            verbose=0,
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
        detailed_rows = []
        subject_rows = []

        for train_index, test_index in logo.split(feature_matrix, labels, groups):
            model = clone(best_estimator)
            x_train = feature_matrix[train_index]
            x_test = feature_matrix[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]
            subject = groups[test_index][0]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            decision_scores = model.decision_function(x_test)
            fold_accuracy = accuracy_score(y_test, y_pred)

            fold_accuracies.append(fold_accuracy)
            subject_rows.append(
                {
                    "subject": subject,
                    "accuracy": fold_accuracy,
                    "n_samples": int(len(y_test)),
                    "n_correct": int(np.sum(y_test == y_pred)),
                    "n_incorrect": int(np.sum(y_test != y_pred)),
                }
            )

            for sample_index, (true_label, predicted_label, decision_score) in enumerate(
                zip(y_test, y_pred, np.asarray(decision_scores))
            ):
                detailed_rows.append(
                    {
                        "subject": subject,
                        "sample_idx": sample_index,
                        "true_label": "truth" if true_label == 0 else "lie",
                        "predicted_label": "truth" if predicted_label == 0 else "lie",
                        "correct": bool(true_label == predicted_label),
                        "decision_score": float(decision_score),
                    }
                )

            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
            decision_scores_all.extend(np.asarray(decision_scores).tolist())

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        auc = roc_auc_score(y_true_all, decision_scores_all)
        f1 = f1_score(y_true_all, y_pred_all)
        losocv_std = float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0

        return {
            "best_cv_score": float(grid_search.best_score_),
            "overall_accuracy": overall_accuracy,
            "auc": auc,
            "f1_score": f1,
            "losocv_std": losocv_std,
            "best_params": best_params,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "subject_rows": subject_rows,
            "detailed_rows": detailed_rows,
            "cv_results": pd.DataFrame(grid_search.cv_results_),
            "final_model": clone(best_estimator).fit(feature_matrix, labels),
            "feature_names": feature_names,
        }

    def save_outputs(self, summary_df, best_result):
        results_path = os.path.join(self.script_directory, "svm_train_new_results.csv")
        summary_df.to_csv(results_path, index=False)

        search_path = os.path.join(self.script_directory, "svm_train_new_grid_search_results.csv")
        best_result["cv_results"].to_csv(search_path, index=False)

        subject_path = os.path.join(self.script_directory, "svm_train_new_subject_results.csv")
        pd.DataFrame(best_result["subject_rows"]).to_csv(subject_path, index=False)

        detailed_path = os.path.join(self.script_directory, "svm_train_new_detailed_results.csv")
        pd.DataFrame(best_result["detailed_rows"]).to_csv(detailed_path, index=False)

        model_path = os.path.join(self.script_directory, "svm_train_new_model.joblib")
        joblib.dump(
            {
                "model": best_result["final_model"],
                "imputer": best_result["imputer"],
                "selected_features": best_result["features"],
                "best_params": best_result["best_params"],
                "best_cv_score": best_result["best_cv_score"],
                "overall_accuracy": best_result["overall_accuracy"],
                "auc": best_result["auc"],
                "f1_score": best_result["f1_score"],
            },
            model_path,
        )

        print("\nSaved outputs:")
        print(f"  - {results_path}")
        print(f"  - {search_path}")
        print(f"  - {subject_path}")
        print(f"  - {detailed_path}")
        print(f"  - {model_path}")

    def run(self):
        print("=== SVM TRAIN NEW (40S) ===")
        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)

        summary_rows = []
        evaluated_results = []

        for index, combination in enumerate(self.combinations, start=1):
            print("\n" + "=" * 80)
            print(f"[{index}/{len(self.combinations)}] Evaluating {combination['name']}")
            print(f"Features ({len(combination['features'])}): {combination['features']}")

            feature_matrix, labels, groups, imputer = self.prepare_matrix(
                dataframe,
                combination["features"],
            )
            result = self.evaluate_combination(
                feature_matrix,
                labels,
                groups,
                combination["features"],
            )
            result["name"] = combination["name"]
            result["features"] = combination["features"]
            result["imputer"] = imputer
            evaluated_results.append(result)

            summary_rows.append(
                {
                    "combination_name": combination["name"],
                    "n_features": len(combination["features"]),
                    "features": ", ".join(combination["features"]),
                    "best_cv_score": result["best_cv_score"],
                    "overall_accuracy": result["overall_accuracy"],
                    "auc": result["auc"],
                    "f1_score": result["f1_score"],
                    "losocv_std": result["losocv_std"],
                    "best_params": str(result["best_params"]),
                    "confusion_matrix": np.array2string(result["confusion_matrix"]),
                }
            )

            print(
                f"Best CV={result['best_cv_score']:.4f}, "
                f"LOSO={result['overall_accuracy']:.4f}, "
                f"AUC={result['auc']:.4f}, "
                f"F1={result['f1_score']:.4f}"
            )
            print(f"Best Params: {result['best_params']}")

        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["overall_accuracy", "best_cv_score", "n_features"],
            ascending=[False, False, True],
        )
        best_name = summary_df.iloc[0]["combination_name"]
        best_result = next(result for result in evaluated_results if result["name"] == best_name)

        print("\n" + "=" * 80)
        print("FINAL BEST RESULT")
        print("=" * 80)
        print(f"Combination: {best_result['name']}")
        print(f"Selected Features: {best_result['features']}")
        print(f"Best CV Accuracy: {best_result['best_cv_score']:.4f}")
        print(f"Final LOSO Accuracy: {best_result['overall_accuracy']:.4f}")
        print(f"LOSOCV Standard Deviation: {best_result['losocv_std']:.4f}")
        print(f"AUC: {best_result['auc']:.4f}")
        print(f"F1 Score: {best_result['f1_score']:.4f}")
        print("Best Parameters:")
        for key, value in best_result["best_params"].items():
            print(f"  - {key}: {value}")
        print("\nConfusion Matrix:")
        print(best_result["confusion_matrix"])
        print("\nClassification Report:")
        print(best_result["classification_report"])

        self.save_outputs(summary_df, best_result)


def main():
    trainer = SVMTrainNew()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()