#!/usr/bin/env python3
"""
Run a staged subset search over the legacy 12 HRV features plus 4 non-linear
features on the 40-second combined dataset. The search uses beam-search style
combinatory elimination from 16 to 12 to 8 to 4 features, with full LOSO grid
search evaluations at each target stage.
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
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class FinalEvalSVMTrainer:
    def __init__(self, data_directory=None, beam_width=6, eval_width=4):
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = data_directory or os.path.normpath(
            os.path.join(self.script_directory, "..", "40s-output")
        )
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
        self.feature_pool = self.legacy_best_12 + self.nonlinear_features
        self.target_sizes = [16, 12, 8, 4]
        self.min_feature_count = min(self.target_sizes)
        self.beam_width = beam_width
        self.eval_width = eval_width
        self.quick_score_cache = {}
        self.full_eval_cache = {}

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

    def prepare_full_matrix(self, dataframe):
        missing_features = [feature for feature in self.feature_pool if feature not in dataframe.columns]
        if missing_features:
            raise ValueError(f"Missing expected HRV features: {missing_features}")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(dataframe[self.feature_pool])
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()

        print(
            f"Prepared dataset: {feature_matrix.shape[0]} samples, "
            f"{feature_matrix.shape[1]} features"
        )
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

    @staticmethod
    def build_quick_pipeline():
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="linear",
                        C=1,
                        class_weight="balanced",
                        random_state=42,
                        cache_size=1024,
                    ),
                ),
            ]
        )

    def subset_to_indices(self, subset):
        return [self.feature_pool.index(feature) for feature in subset]

    def quick_score_subset(self, full_matrix, labels, groups, subset):
        subset = tuple(subset)
        if subset in self.quick_score_cache:
            return self.quick_score_cache[subset]

        logo = LeaveOneGroupOut()
        subset_matrix = full_matrix[:, self.subset_to_indices(subset)]
        scores = cross_val_score(
            self.build_quick_pipeline(),
            subset_matrix,
            labels,
            groups=groups,
            cv=logo,
            scoring="accuracy",
            n_jobs=-1,
        )
        result = {
            "subset": subset,
            "quick_accuracy": float(np.mean(scores)),
            "quick_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "n_features": len(subset),
        }
        self.quick_score_cache[subset] = result
        return result

    def evaluate_full_subset(self, full_matrix, labels, groups, subset):
        subset = tuple(subset)
        if subset in self.full_eval_cache:
            return self.full_eval_cache[subset]

        logo = LeaveOneGroupOut()
        subset_matrix = full_matrix[:, self.subset_to_indices(subset)]
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
        grid_search.fit(subset_matrix, labels, groups=groups)

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

        for train_index, test_index in logo.split(subset_matrix, labels, groups):
            model = clone(best_estimator)
            x_train = subset_matrix[train_index]
            x_test = subset_matrix[test_index]
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

        result = {
            "features": list(subset),
            "n_features": len(subset),
            "best_cv_score": float(grid_search.best_score_),
            "overall_accuracy": accuracy_score(y_true_all, y_pred_all),
            "auc": roc_auc_score(y_true_all, decision_scores_all),
            "f1_score": f1_score(y_true_all, y_pred_all),
            "losocv_std": float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0,
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
            "final_model": clone(best_estimator).fit(subset_matrix, labels),
        }
        self.full_eval_cache[subset] = result
        return result

    def rank_children(self, child_scores):
        return sorted(
            child_scores,
            key=lambda item: (item["quick_accuracy"], -item["quick_std"], item["subset"]),
            reverse=True,
        )

    def run_elimination_search(self, full_matrix, labels, groups):
        current_beam = [tuple(self.feature_pool)]
        elimination_trace = []
        target_evaluations = []

        for stage_size in range(len(self.feature_pool), self.min_feature_count - 1, -1):
            unique_candidates = list(dict.fromkeys(current_beam))

            if stage_size in self.target_sizes:
                ranked_candidates = self.rank_children(
                    [
                        self.quick_score_subset(full_matrix, labels, groups, subset)
                        for subset in unique_candidates
                    ]
                )
                for candidate_rank, candidate in enumerate(
                    ranked_candidates[: min(self.eval_width, len(ranked_candidates))],
                    start=1,
                ):
                    full_result = self.evaluate_full_subset(
                        full_matrix,
                        labels,
                        groups,
                        candidate["subset"],
                    )
                    record = {
                        "stage_size": stage_size,
                        "candidate_rank": candidate_rank,
                        "quick_accuracy": candidate["quick_accuracy"],
                        "quick_std": candidate["quick_std"],
                        **full_result,
                    }
                    target_evaluations.append(record)

                    print(
                        f"Stage {stage_size:2d}, candidate {candidate_rank}: "
                        f"LOSO={full_result['overall_accuracy']:.4f}, "
                        f"CV={full_result['best_cv_score']:.4f}, "
                        f"AUC={full_result['auc']:.4f}, "
                        f"features={list(candidate['subset'])}"
                    )

            if stage_size == self.min_feature_count:
                break

            next_size = stage_size - 1
            child_map = {}
            for subset in unique_candidates:
                for index in range(len(subset)):
                    child = subset[:index] + subset[index + 1 :]
                    if len(child) != next_size or child in child_map:
                        continue
                    child_map[child] = self.quick_score_subset(full_matrix, labels, groups, child)

            ranked_children = self.rank_children(list(child_map.values()))
            for child_rank, child in enumerate(ranked_children, start=1):
                elimination_trace.append(
                    {
                        "from_size": stage_size,
                        "to_size": next_size,
                        "child_rank": child_rank,
                        "n_features": child["n_features"],
                        "quick_accuracy": child["quick_accuracy"],
                        "quick_std": child["quick_std"],
                        "features": ", ".join(child["subset"]),
                    }
                )

            current_beam = [child["subset"] for child in ranked_children[: self.beam_width]]
            print(
                f"Advanced beam from {stage_size} to {next_size} features; "
                f"keeping {len(current_beam)} candidates"
            )

        return target_evaluations, elimination_trace

    def build_best_imputer(self, dataframe, selected_features):
        imputer = SimpleImputer(strategy="median")
        imputer.fit(dataframe[selected_features])
        return imputer

    def save_outputs(self, summary_df, elimination_trace_df, best_result, dataframe):
        results_path = os.path.join(self.script_directory, "svm_train_new_results.csv")
        summary_df.to_csv(results_path, index=False)

        trace_path = os.path.join(self.script_directory, "svm_train_new_elimination_trace.csv")
        elimination_trace_df.to_csv(trace_path, index=False)

        grid_path = os.path.join(self.script_directory, "svm_train_new_grid_search_results.csv")
        best_result["cv_results"].to_csv(grid_path, index=False)

        subject_path = os.path.join(self.script_directory, "svm_train_new_subject_results.csv")
        pd.DataFrame(best_result["subject_rows"]).to_csv(subject_path, index=False)

        detailed_path = os.path.join(self.script_directory, "svm_train_new_detailed_results.csv")
        pd.DataFrame(best_result["detailed_rows"]).to_csv(detailed_path, index=False)

        model_path = os.path.join(self.script_directory, "svm_train_new_model.joblib")
        best_imputer = self.build_best_imputer(dataframe, best_result["features"])
        joblib.dump(
            {
                "model": best_result["final_model"],
                "imputer": best_imputer,
                "selected_features": best_result["features"],
                "best_params": best_result["best_params"],
                "best_cv_score": best_result["best_cv_score"],
                "overall_accuracy": best_result["overall_accuracy"],
                "auc": best_result["auc"],
                "f1_score": best_result["f1_score"],
                "stage_size": best_result["stage_size"],
                "candidate_rank": best_result["candidate_rank"],
            },
            model_path,
        )

        print("\nSaved outputs:")
        print(f"  - {results_path}")
        print(f"  - {trace_path}")
        print(f"  - {grid_path}")
        print(f"  - {subject_path}")
        print(f"  - {detailed_path}")
        print(f"  - {model_path}")

    def run(self):
        print("=== FINAL EVAL SVM TRAIN NEW (40S) ===")
        print(f"Data directory: {self.data_directory}")
        print(f"Feature pool ({len(self.feature_pool)}): {self.feature_pool}")
        print(f"Target subset sizes: {self.target_sizes}")
        print(f"Beam width: {self.beam_width}, full eval width per stage: {self.eval_width}")

        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)
        full_matrix, labels, groups, _ = self.prepare_full_matrix(dataframe)
        target_evaluations, elimination_trace = self.run_elimination_search(
            full_matrix,
            labels,
            groups,
        )

        summary_rows = []
        for result in target_evaluations:
            summary_rows.append(
                {
                    "stage_size": result["stage_size"],
                    "candidate_rank": result["candidate_rank"],
                    "n_features": result["n_features"],
                    "features": ", ".join(result["features"]),
                    "quick_accuracy": result["quick_accuracy"],
                    "quick_std": result["quick_std"],
                    "best_cv_score": result["best_cv_score"],
                    "overall_accuracy": result["overall_accuracy"],
                    "auc": result["auc"],
                    "f1_score": result["f1_score"],
                    "losocv_std": result["losocv_std"],
                    "best_params": str(result["best_params"]),
                    "confusion_matrix": np.array2string(result["confusion_matrix"]),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["overall_accuracy", "best_cv_score", "n_features", "candidate_rank"],
            ascending=[False, False, True, True],
        )
        elimination_trace_df = pd.DataFrame(elimination_trace)

        best_summary = summary_df.iloc[0]
        best_result = next(
            result
            for result in target_evaluations
            if result["stage_size"] == best_summary["stage_size"]
            and result["candidate_rank"] == best_summary["candidate_rank"]
        )

        print("\n" + "=" * 80)
        print("FINAL BEST RESULT")
        print("=" * 80)
        print(f"Stage Size: {best_result['stage_size']}")
        print(f"Candidate Rank: {best_result['candidate_rank']}")
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

        self.save_outputs(summary_df, elimination_trace_df, best_result, dataframe)


def main():
    trainer = FinalEvalSVMTrainer()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()