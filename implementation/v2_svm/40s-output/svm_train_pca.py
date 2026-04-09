#!/usr/bin/env python3
"""
Train a single-stage PCA + SVM model on the full 22-feature HRV set from the
40-second combined dataset. The SVM kernel search matches svm_train_new.py
while also tuning the PCA component count.
"""

import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
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


class SVMTrainPCA:
    def __init__(self, data_directory=None):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or self.script_directory
        self.feature_pool = [
            "nn_count",
            "nn_mean",
            "nn_min",
            "nn_max",
            "sdnn",
            "sdsd",
            "rmssd",
            "pnn20",
            "pnn50",
            "triangular_index",
            "lf_power",
            "hf_power",
            "lf_hf_ratio",
            "lf_norm",
            "hf_norm",
            "ln_hf",
            "lf_peak",
            "hf_peak",
            "sd1",
            "sd2",
            "sd1_sd2_ratio",
            "dfa",
        ]

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

        feature_matrix = dataframe[feature_list].to_numpy(dtype=float)
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()
        return feature_matrix, labels, groups

    @staticmethod
    def build_component_candidates(feature_count, min_train_samples):
        max_components = min(feature_count, min_train_samples)
        if max_components < 1:
            raise ValueError("PCA requires at least one valid component")
        return list(range(1, max_components + 1))

    @staticmethod
    def build_param_grid(component_candidates):
        return [
            {
                "pca__n_components": component_candidates,
                "svm__kernel": ["linear"],
                "svm__C": [0.01, 0.1, 1, 10, 100],
            },
            {
                "pca__n_components": component_candidates,
                "svm__kernel": ["rbf"],
                "svm__C": [0.1, 1, 10, 100],
                "svm__gamma": ["scale", 0.01, 0.1, 1],
            },
            {
                "pca__n_components": component_candidates,
                "svm__kernel": ["poly"],
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", 0.01, 0.1],
                "svm__degree": [2, 3],
                "svm__coef0": [0.0, 1.0],
            },
            {
                "pca__n_components": component_candidates,
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
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(random_state=42)),
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

    @staticmethod
    def build_pca_diagnostics(final_model, feature_names):
        pca = final_model.named_steps["pca"]
        component_labels = [
            f"component_{index + 1}" for index in range(pca.components_.shape[0])
        ]

        explained_variance_df = pd.DataFrame(
            {
                "component": component_labels,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_explained_variance": np.cumsum(
                    pca.explained_variance_ratio_
                ),
            }
        )

        loadings_df = pd.DataFrame(pca.components_.T, columns=component_labels)
        loadings_df.insert(0, "feature", feature_names)
        return explained_variance_df, loadings_df

    @staticmethod
    def compute_metric_std(values):
        valid_values = [value for value in values if not np.isnan(value)]
        if len(valid_values) > 1:
            return float(np.std(valid_values, ddof=1))
        return 0.0

    def evaluate_combination(self, feature_matrix, labels, groups, feature_names):
        logo = LeaveOneGroupOut()
        min_train_samples = min(
            len(train_index)
            for train_index, _ in logo.split(feature_matrix, labels, groups)
        )
        component_candidates = self.build_component_candidates(
            feature_matrix.shape[1],
            min_train_samples,
        )

        grid_search = GridSearchCV(
            estimator=self.build_pipeline(),
            param_grid=self.build_param_grid(component_candidates),
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
            key.replace("svm__", "").replace("pca__", "pca_"): value
            for key, value in grid_search.best_params_.items()
        }

        y_true_all = []
        y_pred_all = []
        decision_scores_all = []
        fold_accuracies = []
        fold_aucs = []
        fold_f1_scores = []
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
            fold_f1 = f1_score(y_test, y_pred, zero_division=0)
            if len(np.unique(y_test)) > 1:
                fold_auc = roc_auc_score(y_test, decision_scores)
            else:
                fold_auc = np.nan

            fold_accuracies.append(fold_accuracy)
            fold_aucs.append(fold_auc)
            fold_f1_scores.append(fold_f1)
            subject_rows.append(
                {
                    "subject": subject,
                    "accuracy": fold_accuracy,
                    "auc": None if np.isnan(fold_auc) else float(fold_auc),
                    "f1_score": float(fold_f1),
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

        final_model = clone(best_estimator).fit(feature_matrix, labels)
        explained_variance_df, loadings_df = self.build_pca_diagnostics(
            final_model,
            feature_names,
        )

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        auc = roc_auc_score(y_true_all, decision_scores_all)
        f1 = f1_score(y_true_all, y_pred_all)
        accuracy_std = self.compute_metric_std(fold_accuracies)
        auc_std = self.compute_metric_std(fold_aucs)
        f1_std = self.compute_metric_std(fold_f1_scores)

        return {
            "best_cv_score": float(grid_search.best_score_),
            "overall_accuracy": overall_accuracy,
            "auc": auc,
            "f1_score": f1,
            "accuracy_std": accuracy_std,
            "auc_std": auc_std,
            "f1_std": f1_std,
            "best_params": best_params,
            "pca_n_components": int(final_model.named_steps["pca"].n_components_),
            "component_candidates": component_candidates,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "subject_rows": subject_rows,
            "detailed_rows": detailed_rows,
            "cv_results": pd.DataFrame(grid_search.cv_results_),
            "final_model": final_model,
            "feature_names": feature_names,
            "explained_variance": explained_variance_df,
            "loadings": loadings_df,
        }

    def save_outputs(self, summary_df, best_result):
        results_path = os.path.join(self.script_directory, "svm_train_pca_results.csv")
        summary_df.to_csv(results_path, index=False)

        search_path = os.path.join(self.script_directory, "svm_train_pca_grid_search_results.csv")
        best_result["cv_results"].to_csv(search_path, index=False)

        subject_path = os.path.join(self.script_directory, "svm_train_pca_subject_results.csv")
        pd.DataFrame(best_result["subject_rows"]).to_csv(subject_path, index=False)

        detailed_path = os.path.join(self.script_directory, "svm_train_pca_detailed_results.csv")
        pd.DataFrame(best_result["detailed_rows"]).to_csv(detailed_path, index=False)

        explained_variance_path = os.path.join(
            self.script_directory,
            "svm_train_pca_explained_variance.csv",
        )
        best_result["explained_variance"].to_csv(explained_variance_path, index=False)

        loadings_path = os.path.join(self.script_directory, "svm_train_pca_loadings.csv")
        best_result["loadings"].to_csv(loadings_path, index=False)

        model_path = os.path.join(self.script_directory, "svm_train_pca_model.joblib")
        joblib.dump(
            {
                "model": best_result["final_model"],
                "selected_features": best_result["features"],
                "best_params": best_result["best_params"],
                "best_cv_score": best_result["best_cv_score"],
                "overall_accuracy": best_result["overall_accuracy"],
                "auc": best_result["auc"],
                "f1_score": best_result["f1_score"],
                "pca_n_components": best_result["pca_n_components"],
                "feature_names": best_result["feature_names"],
            },
            model_path,
        )

        print("\nSaved outputs:")
        print(f"  - {results_path}")
        print(f"  - {search_path}")
        print(f"  - {subject_path}")
        print(f"  - {detailed_path}")
        print(f"  - {explained_variance_path}")
        print(f"  - {loadings_path}")
        print(f"  - {model_path}")

    def run(self):
        print("=== SVM TRAIN PCA (40S) ===")
        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)

        print("\n" + "=" * 80)
        print("Single-stage PCA evaluation")
        print(f"Features ({len(self.feature_pool)}): {self.feature_pool}")

        feature_matrix, labels, groups = self.prepare_matrix(
            dataframe,
            self.feature_pool,
        )
        best_result = self.evaluate_combination(
            feature_matrix,
            labels,
            groups,
            self.feature_pool,
        )
        best_result["name"] = "all_features_pca"
        best_result["features"] = self.feature_pool

        summary_df = pd.DataFrame(
            [
                {
                    "combination_name": best_result["name"],
                    "n_features": len(self.feature_pool),
                    "features": ", ".join(self.feature_pool),
                    "pca_n_components": best_result["pca_n_components"],
                    "component_candidates": str(best_result["component_candidates"]),
                    "best_cv_score": best_result["best_cv_score"],
                    "overall_accuracy": best_result["overall_accuracy"],
                    "auc": best_result["auc"],
                    "f1_score": best_result["f1_score"],
                    "accuracy_std": best_result["accuracy_std"],
                    "auc_std": best_result["auc_std"],
                    "f1_std": best_result["f1_std"],
                    "best_params": str(best_result["best_params"]),
                    "confusion_matrix": np.array2string(best_result["confusion_matrix"]),
                }
            ]
        )

        print(
            f"Best CV={best_result['best_cv_score']:.4f}, "
            f"LOSO={best_result['overall_accuracy']:.4f} +- {best_result['accuracy_std']:.4f}, "
            f"AUC={best_result['auc']:.4f} +- {best_result['auc_std']:.4f}, "
            f"F1={best_result['f1_score']:.4f} +- {best_result['f1_std']:.4f}"
        )
        print(f"Best Params: {best_result['best_params']}")

        print("\n" + "=" * 80)
        print("FINAL BEST RESULT")
        print("=" * 80)
        print(f"Combination: {best_result['name']}")
        print(f"Selected Features: {best_result['features']}")
        print(f"PCA Components: {best_result['pca_n_components']}")
        print(f"Best CV Accuracy: {best_result['best_cv_score']:.4f}")
        print(
            f"Final LOSO Accuracy: {best_result['overall_accuracy']:.4f} "
            f"+- {best_result['accuracy_std']:.4f}"
        )
        print(f"AUC: {best_result['auc']:.4f} +- {best_result['auc_std']:.4f}")
        print(f"F1 Score: {best_result['f1_score']:.4f} +- {best_result['f1_std']:.4f}")
        print("Best Parameters:")
        for key, value in best_result["best_params"].items():
            print(f"  - {key}: {value}")
        print("\nConfusion Matrix:")
        print(best_result["confusion_matrix"])
        print("\nClassification Report:")
        print(best_result["classification_report"])

        self.save_outputs(summary_df, best_result)


def main():
    trainer = SVMTrainPCA()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()