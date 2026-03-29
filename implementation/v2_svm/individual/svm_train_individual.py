#!/usr/bin/env python3
"""
Train individual subject-specific SVM models on the 40-second combined HRV
dataset using the same staged feature-selection and hyperparameter optimization
flow as the final evaluation pipeline, but evaluated within each subject across
sessions.
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


class IndividualSVMTrainer:
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

    def reset_subject_caches(self):
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
        return combined_df

    def prepare_subject_matrix(self, subject_df):
        missing_features = [feature for feature in self.feature_pool if feature not in subject_df.columns]
        if missing_features:
            raise ValueError(f"Missing expected HRV features: {missing_features}")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(subject_df[self.feature_pool])
        labels = subject_df["binary_label"].to_numpy()
        groups = subject_df["condition"].to_numpy()
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

    def evaluate_full_subset(self, subject, full_matrix, labels, groups, subset):
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
        session_rows = []
        detailed_rows = []

        for train_index, test_index in logo.split(subset_matrix, labels, groups):
            model = clone(best_estimator)
            x_train = subset_matrix[train_index]
            x_test = subset_matrix[test_index]
            y_train = labels[train_index]
            y_test = labels[test_index]
            session = groups[test_index][0]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            decision_scores = model.decision_function(x_test)
            fold_accuracy = accuracy_score(y_test, y_pred)

            fold_accuracies.append(fold_accuracy)
            session_rows.append(
                {
                    "subject": subject,
                    "session": session,
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
                        "session": session,
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
            "subject": subject,
            "features": list(subset),
            "n_features": len(subset),
            "best_cv_score": float(grid_search.best_score_),
            "overall_accuracy": accuracy_score(y_true_all, y_pred_all),
            "auc": roc_auc_score(y_true_all, decision_scores_all),
            "f1_score": f1_score(y_true_all, y_pred_all),
            "session_std": float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0,
            "best_params": best_params,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "session_rows": session_rows,
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

    def run_subject_search(self, subject, full_matrix, labels, groups):
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
                        subject,
                        full_matrix,
                        labels,
                        groups,
                        candidate["subset"],
                    )
                    record = {
                        "subject": subject,
                        "stage_size": stage_size,
                        "candidate_rank": candidate_rank,
                        "quick_accuracy": candidate["quick_accuracy"],
                        "quick_std": candidate["quick_std"],
                        **full_result,
                    }
                    target_evaluations.append(record)

                    print(
                        f"Stage {stage_size:2d}, candidate {candidate_rank}: "
                        f"ACC={full_result['overall_accuracy']:.4f}, "
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
                        "subject": subject,
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

    def save_subject_model(self, result, subject_df):
        model_path = os.path.join(self.script_directory, f"{result['subject']}_model.joblib")
        imputer = SimpleImputer(strategy="median")
        imputer.fit(subject_df[result["features"]])
        joblib.dump(
            {
                "model": result["final_model"],
                "imputer": imputer,
                "selected_features": result["features"],
                "best_params": result["best_params"],
                "best_cv_score": result["best_cv_score"],
                "overall_accuracy": result["overall_accuracy"],
                "auc": result["auc"],
                "f1_score": result["f1_score"],
                "subject": result["subject"],
                "sessions": result["n_sessions"],
                "stage_size": result["stage_size"],
                "candidate_rank": result["candidate_rank"],
            },
            model_path,
        )
        return model_path

    def save_outputs(self, summary_df, session_df, detailed_df, all_cv_results, trace_df):
        results_path = os.path.join(self.script_directory, "individual_train_results.csv")
        summary_df.to_csv(results_path, index=False)

        trace_path = os.path.join(self.script_directory, "individual_train_elimination_trace.csv")
        trace_df.to_csv(trace_path, index=False)

        session_path = os.path.join(self.script_directory, "individual_train_session_results.csv")
        session_df.to_csv(session_path, index=False)

        detailed_path = os.path.join(self.script_directory, "individual_train_detailed_results.csv")
        detailed_df.to_csv(detailed_path, index=False)

        grid_path = os.path.join(self.script_directory, "individual_train_grid_search_results.csv")
        all_cv_results.to_csv(grid_path, index=False)

        print("\nSaved outputs:")
        print(f"  - {results_path}")
        print(f"  - {trace_path}")
        print(f"  - {session_path}")
        print(f"  - {detailed_path}")
        print(f"  - {grid_path}")

    def run(self):
        print("=== INDIVIDUAL SVM TRAINING (CROSS-SESSION) ===")
        print(f"Data directory: {self.data_directory}")
        print(f"Feature pool ({len(self.feature_pool)}): {self.feature_pool}")
        print(f"Target subset sizes: {self.target_sizes}")
        print(f"Beam width: {self.beam_width}, full eval width per stage: {self.eval_width}")

        file_metadata = self.discover_files()
        dataframe = self.load_data(file_metadata)

        summary_rows = []
        all_session_rows = []
        all_detailed_rows = []
        all_cv_results = []
        all_trace_rows = []

        for subject in sorted(pd.unique(dataframe["subject"])):
            subject_df = dataframe[dataframe["subject"] == subject].copy()
            full_matrix, labels, groups, _ = self.prepare_subject_matrix(subject_df)
            unique_sessions = sorted(pd.unique(groups))
            if len(unique_sessions) < 2:
                raise ValueError(f"Subject {subject} has fewer than 2 sessions: {unique_sessions}")

            self.reset_subject_caches()

            print("\n" + "=" * 80)
            print(f"Training subject: {subject}")
            print(f"Samples: {len(subject_df)}, sessions: {unique_sessions}")

            target_evaluations, elimination_trace = self.run_subject_search(
                subject,
                full_matrix,
                labels,
                groups,
            )
            all_trace_rows.extend(elimination_trace)

            subject_summary = pd.DataFrame(
                [
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
                        "session_std": result["session_std"],
                        "best_params": str(result["best_params"]),
                        "confusion_matrix": np.array2string(result["confusion_matrix"]),
                    }
                    for result in target_evaluations
                ]
            ).sort_values(
                by=["overall_accuracy", "best_cv_score", "n_features", "candidate_rank"],
                ascending=[False, False, True, True],
            )

            best_summary = subject_summary.iloc[0]
            best_result = next(
                result
                for result in target_evaluations
                if result["stage_size"] == best_summary["stage_size"]
                and result["candidate_rank"] == best_summary["candidate_rank"]
            )
            best_result["n_sessions"] = len(unique_sessions)
            best_result["n_samples"] = int(len(subject_df))

            model_path = self.save_subject_model(best_result, subject_df)

            summary_rows.append(
                {
                    "subject": subject,
                    "n_sessions": best_result["n_sessions"],
                    "n_samples": best_result["n_samples"],
                    "stage_size": best_result["stage_size"],
                    "candidate_rank": best_result["candidate_rank"],
                    "n_features": best_result["n_features"],
                    "features": ", ".join(best_result["features"]),
                    "quick_accuracy": best_result["quick_accuracy"],
                    "quick_std": best_result["quick_std"],
                    "best_cv_score": best_result["best_cv_score"],
                    "overall_accuracy": best_result["overall_accuracy"],
                    "auc": best_result["auc"],
                    "f1_score": best_result["f1_score"],
                    "session_std": best_result["session_std"],
                    "best_params": str(best_result["best_params"]),
                    "confusion_matrix": np.array2string(best_result["confusion_matrix"]),
                    "model_path": model_path,
                }
            )
            all_session_rows.extend(best_result["session_rows"])
            all_detailed_rows.extend(best_result["detailed_rows"])

            cv_result_df = best_result["cv_results"].copy()
            cv_result_df.insert(0, "subject", subject)
            cv_result_df.insert(1, "stage_size", best_result["stage_size"])
            cv_result_df.insert(2, "candidate_rank", best_result["candidate_rank"])
            all_cv_results.append(cv_result_df)

            print(
                f"Accuracy={best_result['overall_accuracy']:.4f}, "
                f"AUC={best_result['auc']:.4f}, "
                f"F1={best_result['f1_score']:.4f}, "
                f"Stage={best_result['stage_size']}, "
                f"Features={best_result['features']}, "
                f"Params={best_result['best_params']}"
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            by=["overall_accuracy", "auc", "subject"],
            ascending=[False, False, True],
        )
        session_df = pd.DataFrame(all_session_rows)
        detailed_df = pd.DataFrame(all_detailed_rows)
        all_cv_results_df = pd.concat(all_cv_results, ignore_index=True)
        trace_df = pd.DataFrame(all_trace_rows)

        print("\n" + "=" * 80)
        print("OVERALL INDIVIDUAL RESULTS")
        print("=" * 80)
        print(
            summary_df[
                [
                    "subject",
                    "stage_size",
                    "n_features",
                    "overall_accuracy",
                    "auc",
                    "f1_score",
                    "best_params",
                ]
            ].to_string(index=False)
        )
        print("\nMean Accuracy:", f"{summary_df['overall_accuracy'].mean():.4f}")
        print("Mean AUC:", f"{summary_df['auc'].mean():.4f}")
        print("Mean F1:", f"{summary_df['f1_score'].mean():.4f}")

        self.save_outputs(summary_df, session_df, detailed_df, all_cv_results_df, trace_df)


def main():
    trainer = IndividualSVMTrainer()
    try:
        trainer.run()
    except Exception as error:
        print(f"Error: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()