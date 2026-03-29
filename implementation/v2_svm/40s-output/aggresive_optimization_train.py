#!/usr/bin/env python3
"""
Aggressively optimize 40-second HRV SVM training by combining feature ranking,
feature-count search from 8 down to 1, and kernel hyperparameter search under
Leave-One-Subject-Out cross-validation.
"""

import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
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


class AggresiveOptimizationTrainer:
    def __init__(self, data_directory=None):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or self.script_directory
        self.time_domain_features = [
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
        ]
        self.frequency_domain_features = [
            "lf_power",
            "hf_power",
            "lf_hf_ratio",
            "lf_norm",
            "hf_norm",
            "ln_hf",
            "lf_peak",
            "hf_peak",
        ]
        self.nonlinear_features = ["sd1", "sd2", "sd1_sd2_ratio", "dfa"]
        self.feature_columns = (
            self.time_domain_features
            + self.frequency_domain_features
            + self.nonlinear_features
        )
        self.min_feature_count = 1
        self.max_feature_count = 8

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
        available_features = [
            feature for feature in self.feature_columns if feature in dataframe.columns
        ]
        missing_features = [
            feature for feature in self.feature_columns if feature not in dataframe.columns
        ]
        if missing_features:
            raise ValueError(f"Missing expected HRV features: {missing_features}")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(dataframe[available_features])
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()

        print(f"Using all {len(available_features)} HRV features")
        print(
            f"Final dataset: {feature_matrix.shape[0]} samples, "
            f"{feature_matrix.shape[1]} features"
        )
        return feature_matrix, labels, groups, available_features, imputer

    @staticmethod
    def _normalize_scores(values):
        values = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        max_value = np.max(values) if values.size else 0.0
        if max_value <= 0:
            return np.zeros_like(values, dtype=float)
        return values / max_value

    def rank_features(self, feature_matrix, labels, feature_names):
        print("\n" + "=" * 60)
        print("FEATURE RANKING")
        print("=" * 60)

        f_selector = SelectKBest(score_func=f_classif, k="all")
        f_selector.fit(feature_matrix, labels)
        f_scores = np.nan_to_num(f_selector.scores_, nan=0.0)
        f_pvalues = np.nan_to_num(f_selector.pvalues_, nan=1.0)

        mi_scores = mutual_info_classif(feature_matrix, labels, random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)

        random_forest = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )
        random_forest.fit(feature_matrix, labels)
        rf_scores = np.nan_to_num(random_forest.feature_importances_, nan=0.0)

        scaled_features = StandardScaler().fit_transform(feature_matrix)
        rfe = RFE(
            estimator=SVC(kernel="linear", class_weight="balanced"),
            n_features_to_select=1,
            step=1,
        )
        rfe.fit(scaled_features, labels)
        rfe_ranking = rfe.ranking_
        rfe_scores = 1.0 / rfe_ranking.astype(float)

        ranking_df = pd.DataFrame(
            {
                "feature": feature_names,
                "f_score": f_scores,
                "f_pvalue": f_pvalues,
                "mutual_info": mi_scores,
                "random_forest": rf_scores,
                "svm_rfe_inverse_rank": rfe_scores,
            }
        )

        ranking_df["f_score_norm"] = self._normalize_scores(ranking_df["f_score"].to_numpy())
        ranking_df["mutual_info_norm"] = self._normalize_scores(ranking_df["mutual_info"].to_numpy())
        ranking_df["random_forest_norm"] = self._normalize_scores(ranking_df["random_forest"].to_numpy())
        ranking_df["svm_rfe_norm"] = self._normalize_scores(ranking_df["svm_rfe_inverse_rank"].to_numpy())
        ranking_df["consensus_score"] = ranking_df[
            [
                "f_score_norm",
                "mutual_info_norm",
                "random_forest_norm",
                "svm_rfe_norm",
            ]
        ].mean(axis=1)
        ranking_df["consensus_rank"] = ranking_df["consensus_score"].rank(
            ascending=False,
            method="dense",
        ).astype(int)
        ranking_df = ranking_df.sort_values(
            by=["consensus_rank", "consensus_score", "f_score"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

        print("Top ranked features:")
        for index, row in ranking_df.head(self.max_feature_count).iterrows():
            print(
                f"  {index + 1:2d}. {row['feature']:<20} "
                f"score={row['consensus_score']:.4f}"
            )

        return ranking_df

    def build_param_grid(self):
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

    def evaluate_subset(self, feature_matrix, labels, groups, selected_features, feature_names):
        selected_indices = [feature_names.index(feature) for feature in selected_features]
        selected_matrix = feature_matrix[:, selected_indices]
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

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.build_param_grid(),
            cv=logo,
            scoring="accuracy",
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            verbose=0,
        )
        grid_search.fit(selected_matrix, labels, groups=groups)

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

        for train_index, test_index in logo.split(selected_matrix, labels, groups):
            model = clone(best_estimator)
            x_train = selected_matrix[train_index]
            x_test = selected_matrix[test_index]
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

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        overall_auc = roc_auc_score(y_true_all, decision_scores_all)
        overall_f1 = f1_score(y_true_all, y_pred_all)
        losocv_std = float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0

        return {
            "feature_count": len(selected_features),
            "selected_features": selected_features,
            "best_cv_score": float(grid_search.best_score_),
            "overall_accuracy": overall_accuracy,
            "auc": overall_auc,
            "f1_score": overall_f1,
            "losocv_std": losocv_std,
            "best_params": best_params,
            "subject_results": subject_results,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "cv_results": pd.DataFrame(grid_search.cv_results_),
            "final_model": clone(best_estimator).fit(selected_matrix, labels),
        }

    def optimize_and_train(self, feature_matrix, labels, groups, ranking_df, feature_names):
        print("\n" + "=" * 60)
        print("AGGRESSIVE OPTIMIZATION")
        print("=" * 60)
        print(
            f"Feature-count search: {self.max_feature_count} down to {self.min_feature_count}"
        )

        ranked_features = ranking_df["feature"].tolist()
        all_results = []
        best_result = None

        for feature_count in range(self.max_feature_count, self.min_feature_count - 1, -1):
            selected_features = ranked_features[:feature_count]
            result = self.evaluate_subset(
                feature_matrix,
                labels,
                groups,
                selected_features,
                feature_names,
            )

            all_results.append(
                {
                    "feature_count": result["feature_count"],
                    "best_cv_score": result["best_cv_score"],
                    "overall_accuracy": result["overall_accuracy"],
                    "auc": result["auc"],
                    "f1_score": result["f1_score"],
                    "losocv_std": result["losocv_std"],
                    "selected_features": ", ".join(result["selected_features"]),
                    "best_params": str(result["best_params"]),
                }
            )

            print(
                f"Top-{feature_count:2d} features: "
                f"CV={result['best_cv_score']:.4f}, "
                f"LOSO={result['overall_accuracy']:.4f}, "
                f"AUC={result['auc']:.4f}, "
                f"F1={result['f1_score']:.4f}"
            )

            if best_result is None:
                best_result = result
                continue

            if result["best_cv_score"] > best_result["best_cv_score"]:
                best_result = result
            elif result["best_cv_score"] == best_result["best_cv_score"]:
                if result["feature_count"] < best_result["feature_count"]:
                    best_result = result

        cv_results = pd.DataFrame(all_results).sort_values(
            by=["best_cv_score", "feature_count"],
            ascending=[False, True],
        )

        return {
            "overall_accuracy": best_result["overall_accuracy"],
            "best_cv_score": best_result["best_cv_score"],
            "best_params": best_result["best_params"],
            "selected_features": best_result["selected_features"],
            "subject_results": best_result["subject_results"],
            "confusion_matrix": best_result["confusion_matrix"],
            "classification_report": best_result["classification_report"],
            "cv_results": cv_results,
            "search_details": all_results,
            "final_model": best_result["final_model"],
            "best_feature_count": best_result["feature_count"],
            "auc": best_result["auc"],
            "f1_score": best_result["f1_score"],
            "losocv_std": best_result["losocv_std"],
        }

    def save_results(self, ranking_df, results, imputer):
        ranking_path = os.path.join(
            self.script_directory,
            "aggresive_optimization_feature_ranking.csv",
        )
        ranking_df.to_csv(ranking_path, index=False)

        results_path = os.path.join(
            self.script_directory,
            "aggresive_optimization_train_results.csv",
        )
        pd.DataFrame(
            [
                {
                    "loso_accuracy": results["overall_accuracy"],
                    "best_cv_score": results["best_cv_score"],
                    "best_feature_count": results["best_feature_count"],
                    "losocv_standard_deviation": results["losocv_std"],
                    "auc": results["auc"],
                    "f1_score": results["f1_score"],
                    "selected_features": ", ".join(results["selected_features"]),
                    "best_params": str(results["best_params"]),
                    "confusion_matrix": np.array2string(results["confusion_matrix"]),
                }
            ]
        ).to_csv(results_path, index=False)

        search_path = os.path.join(
            self.script_directory,
            "aggresive_optimization_search_results.csv",
        )
        pd.DataFrame(results["search_details"]).to_csv(search_path, index=False)

        subject_path = os.path.join(
            self.script_directory,
            "aggresive_optimization_subject_results.csv",
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

        model_path = os.path.join(
            self.script_directory,
            "aggresive_optimization_train_model.joblib",
        )
        artifact = {
            "model": results["final_model"],
            "imputer": imputer,
            "selected_features": results["selected_features"],
            "best_params": results["best_params"],
        }
        joblib.dump(artifact, model_path)

        print("\nSaved outputs:")
        print(f"  - {ranking_path}")
        print(f"  - {results_path}")
        print(f"  - {search_path}")
        print(f"  - {subject_path}")
        print(f"  - {model_path}")

    def run(self):
        print("=== AGGRESIVE OPTIMIZATION TRAINING (40S) ===")
        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)
        feature_matrix, labels, groups, feature_names, imputer = self.prepare_features(dataframe)
        ranking_df = self.rank_features(feature_matrix, labels, feature_names)
        results = self.optimize_and_train(
            feature_matrix,
            labels,
            groups,
            ranking_df,
            feature_names,
        )

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Best CV Accuracy: {results['best_cv_score']:.4f}")
        print(f"Final LOSO Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Best Feature Count: {results['best_feature_count']}")
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

        self.save_results(ranking_df, results, imputer)


def main():
    trainer = AggresiveOptimizationTrainer()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()