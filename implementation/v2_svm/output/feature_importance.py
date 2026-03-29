#!/usr/bin/env python3
"""
HRV-based lie detection with feature importance analysis and joint SVM optimization.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


class FeatureImportanceHRVDetector:
    def __init__(self, data_directory=None):
        """Initialize the HRV lie detection system."""
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

        self.all_data = None
        self.feature_importance_results = {}

    def parse_filename(self, filename):
        """Parse current v2 combined CSV filename format."""
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
        """Discover current v2 combined feature files."""
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
        """Load all v2 combined CSV files into a single dataset."""
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
        """Prepare feature matrix from the combined v2 dataset."""
        available_features = [
            feature for feature in self.feature_columns if feature in dataframe.columns
        ]
        missing_features = [
            feature for feature in self.feature_columns if feature not in dataframe.columns
        ]

        if not available_features:
            raise ValueError("No expected feature columns were found in the combined dataset")

        if missing_features:
            print(f"Warning: Missing features {missing_features}")

        print(f"Using {len(available_features)} features from combined v2 outputs")

        imputer = SimpleImputer(strategy="median")
        feature_matrix = imputer.fit_transform(dataframe[available_features])
        labels = dataframe["binary_label"].to_numpy()
        groups = dataframe["subject"].to_numpy()

        print(
            f"Final dataset: {feature_matrix.shape[0]} samples, "
            f"{feature_matrix.shape[1]} features"
        )
        print(
            f"Label distribution: Truth={np.sum(labels == 0)}, "
            f"Lie={np.sum(labels == 1)}"
        )
        print(f"Features: {available_features}")

        return feature_matrix, labels, groups, available_features

    @staticmethod
    def _normalize_scores(values):
        """Normalize a score array to the range [0, 1]."""
        values = np.nan_to_num(values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        max_value = np.max(values) if values.size else 0.0
        if max_value <= 0:
            return np.zeros_like(values, dtype=float)
        return values / max_value

    def analyze_feature_importance(self, feature_matrix, labels, feature_names):
        """Analyze feature importance using multiple ranking methods."""
        print(f"\n{'=' * 60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'=' * 60}")

        importance_results = {}

        print("\n1. Univariate F-score Analysis...")
        f_selector = SelectKBest(score_func=f_classif, k="all")
        f_selector.fit(feature_matrix, labels)
        f_scores = np.nan_to_num(f_selector.scores_, nan=0.0)
        f_pvalues = np.nan_to_num(f_selector.pvalues_, nan=1.0)
        importance_results["f_score"] = {
            "scores": f_scores,
            "pvalues": f_pvalues,
            "ranking": np.argsort(f_scores)[::-1],
        }

        print("Top 10 features by F-score:")
        for rank, index in enumerate(importance_results["f_score"]["ranking"][:10], start=1):
            print(
                f"  {rank:2d}. {feature_names[index]:<20} "
                f"F={f_scores[index]:.3f} p={f_pvalues[index]:.4f}"
            )

        print("\n2. Mutual Information Analysis...")
        mi_scores = mutual_info_classif(feature_matrix, labels, random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        importance_results["mutual_info"] = {
            "scores": mi_scores,
            "ranking": np.argsort(mi_scores)[::-1],
        }

        print("Top 10 features by Mutual Information:")
        for rank, index in enumerate(
            importance_results["mutual_info"]["ranking"][:10],
            start=1,
        ):
            print(f"  {rank:2d}. {feature_names[index]:<20} MI={mi_scores[index]:.4f}")

        print("\n3. Random Forest Feature Importance...")
        random_forest = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )
        random_forest.fit(feature_matrix, labels)
        rf_scores = np.nan_to_num(random_forest.feature_importances_, nan=0.0)
        importance_results["random_forest"] = {
            "scores": rf_scores,
            "ranking": np.argsort(rf_scores)[::-1],
        }

        print("Top 10 features by Random Forest:")
        for rank, index in enumerate(
            importance_results["random_forest"]["ranking"][:10],
            start=1,
        ):
            print(
                f"  {rank:2d}. {feature_names[index]:<20} "
                f"Importance={rf_scores[index]:.4f}"
            )

        print("\n4. SVM-RFE Analysis...")
        scaled_features = StandardScaler().fit_transform(feature_matrix)
        rfe = RFE(
            estimator=SVC(kernel="linear", class_weight="balanced"),
            n_features_to_select=1,
            step=1,
        )
        rfe.fit(scaled_features, labels)
        rfe_ranking = rfe.ranking_
        importance_results["svm_rfe"] = {
            "ranking": rfe_ranking,
            "feature_ranking": np.argsort(rfe_ranking),
        }

        print("Top 10 features by SVM-RFE:")
        for rank, index in enumerate(
            importance_results["svm_rfe"]["feature_ranking"][:10],
            start=1,
        ):
            print(f"  {rank:2d}. {feature_names[index]:<20} Rank={rfe_ranking[index]}")

        print("\n5. Correlation Analysis...")
        correlations = []
        for column_index in range(feature_matrix.shape[1]):
            column = feature_matrix[:, column_index]
            if np.std(column) == 0:
                correlations.append(0.0)
            else:
                correlations.append(np.corrcoef(column, labels)[0, 1])
        correlations = np.nan_to_num(np.array(correlations), nan=0.0)
        abs_correlations = np.abs(correlations)
        importance_results["correlation"] = {
            "correlations": correlations,
            "abs_correlations": abs_correlations,
            "ranking": np.argsort(abs_correlations)[::-1],
        }

        print("Top 10 features by Absolute Correlation:")
        for rank, index in enumerate(
            importance_results["correlation"]["ranking"][:10],
            start=1,
        ):
            print(
                f"  {rank:2d}. {feature_names[index]:<20} "
                f"|r|={abs_correlations[index]:.4f} r={correlations[index]:.4f}"
            )

        self.feature_importance_results = importance_results
        return importance_results

    def create_consensus_ranking(self, importance_results, feature_names):
        """Create a consensus ranking across all importance methods."""
        print(f"\n{'=' * 60}")
        print("CONSENSUS FEATURE RANKING")
        print(f"{'=' * 60}")

        n_features = len(feature_names)
        normalized_scores = {
            "f_score": self._normalize_scores(importance_results["f_score"]["scores"]),
            "mutual_info": self._normalize_scores(
                importance_results["mutual_info"]["scores"]
            ),
            "random_forest": self._normalize_scores(
                importance_results["random_forest"]["scores"]
            ),
            "svm_rfe": (n_features - importance_results["svm_rfe"]["ranking"] + 1)
            / n_features,
            "correlation": self._normalize_scores(
                importance_results["correlation"]["abs_correlations"]
            ),
        }

        consensus_scores = np.zeros(n_features, dtype=float)
        for method_scores in normalized_scores.values():
            consensus_scores += method_scores
        consensus_scores /= len(normalized_scores)
        consensus_ranking = np.argsort(consensus_scores)[::-1]

        print("CONSENSUS TOP 10 FEATURES:")
        print(
            f"{'Rank':<4} {'Feature':<20} {'Consensus':<10} {'F-score':<8} "
            f"{'MI':<8} {'RF':<8} {'RFE':<8} {'Corr':<8}"
        )
        print("-" * 88)

        for rank, index in enumerate(consensus_ranking[:10], start=1):
            print(
                f"{rank:<4} {feature_names[index]:<20} {consensus_scores[index]:.3f}      "
                f"{normalized_scores['f_score'][index]:.3f}   "
                f"{normalized_scores['mutual_info'][index]:.3f}   "
                f"{normalized_scores['random_forest'][index]:.3f}   "
                f"{normalized_scores['svm_rfe'][index]:.3f}   "
                f"{normalized_scores['correlation'][index]:.3f}"
            )

        return consensus_ranking, consensus_scores

    @staticmethod
    def build_feature_count_grid(total_features):
        """Build a compact but meaningful grid of feature subset sizes."""
        candidate_sizes = [4, 6, 8, 10, 12, 15, total_features]
        return sorted({size for size in candidate_sizes if 1 <= size <= total_features})

    def optimize_model(self, feature_matrix, labels, groups, feature_names):
        """Jointly optimize feature count and SVM kernel hyperparameters."""
        print(f"\n{'=' * 60}")
        print("JOINT FEATURE + KERNEL GRID SEARCH")
        print(f"{'=' * 60}")

        feature_count_grid = self.build_feature_count_grid(len(feature_names))
        print(f"Feature count grid: {feature_count_grid}")

        logo = LeaveOneGroupOut()
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("svm", SVC(class_weight="balanced", random_state=42)),
            ]
        )

        param_grid = [
            {
                "selector__k": feature_count_grid,
                "svm__kernel": ["linear"],
                "svm__C": [0.1, 1, 10],
            },
            {
                "selector__k": feature_count_grid,
                "svm__kernel": ["rbf"],
                "svm__C": [0.1, 1, 10],
                "svm__gamma": ["scale", 0.01, 0.1],
            },
            {
                "selector__k": feature_count_grid,
                "svm__kernel": ["poly"],
                "svm__C": [0.1, 1],
                "svm__gamma": ["scale"],
                "svm__degree": [2, 3],
                "svm__coef0": [0.0, 1.0],
            },
        ]

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

        print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")

        best_estimator = grid_search.best_estimator_
        selected_indices = np.flatnonzero(best_estimator.named_steps["selector"].get_support())
        selected_features = [feature_names[index] for index in selected_indices]

        print("Selected features from best model:")
        for rank, feature_name in enumerate(selected_features, start=1):
            print(f"  {rank:2d}. {feature_name}")

        cv_results = pd.DataFrame(grid_search.cv_results_)
        top_models = cv_results.sort_values("rank_test_score").head(10)
        print("\nTop 10 grid-search configurations:")
        display_columns = [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "param_selector__k",
            "param_svm__kernel",
            "param_svm__C",
            "param_svm__gamma",
            "param_svm__degree",
            "param_svm__coef0",
        ]
        print(top_models[[column for column in display_columns if column in top_models.columns]].to_string(index=False))

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
                f"Subject {test_subject}: {subject_results[test_subject]['accuracy']:.4f} "
                f"({subject_results[test_subject]['n_samples']} samples)"
            )

        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        return {
            "overall_accuracy": overall_accuracy,
            "best_cv_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "selected_feature_indices": selected_indices.tolist(),
            "selected_features": selected_features,
            "subject_results": subject_results,
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all),
            "classification_report": classification_report(
                y_true_all,
                y_pred_all,
                target_names=["Truth", "Lie"],
            ),
            "cv_results": cv_results,
        }

    def run_complete_analysis(self):
        """Run the complete feature-importance and optimization workflow."""
        print("=== HRV LIE DETECTION WITH FEATURE IMPORTANCE ANALYSIS ===")

        print("\n1. Loading data...")
        file_metadata = self.discover_files()
        self.all_data = self.load_data(file_metadata)

        print("\n2. Preparing features...")
        feature_matrix, labels, groups, feature_names = self.prepare_features(self.all_data)

        print("\n3. Analyzing feature importance...")
        importance_results = self.analyze_feature_importance(
            feature_matrix,
            labels,
            feature_names,
        )

        print("\n4. Creating consensus ranking...")
        consensus_ranking, consensus_scores = self.create_consensus_ranking(
            importance_results,
            feature_names,
        )

        print("\n5. Jointly optimizing feature count and kernel...")
        final_results = self.optimize_model(
            feature_matrix,
            labels,
            groups,
            feature_names,
        )

        print(f"\n{'=' * 80}")
        print("FINAL OPTIMIZED RESULTS")
        print(f"{'=' * 80}")
        print(f"Best CV Accuracy: {final_results['best_cv_score']:.4f}")
        print(f"Final LOSO Accuracy: {final_results['overall_accuracy']:.4f}")
        print(f"Selected Features ({len(final_results['selected_features'])}):")
        for feature_name in final_results["selected_features"]:
            print(f"  - {feature_name}")
        print("Best Parameters:")
        for key, value in final_results["best_params"].items():
            print(f"  - {key}: {value}")
        print("\nConfusion Matrix:")
        print(final_results["confusion_matrix"])

        self.save_importance_results(
            importance_results,
            consensus_scores,
            consensus_ranking,
            final_results,
            feature_names,
        )

        return final_results

    def save_importance_results(
        self,
        importance_results,
        consensus_scores,
        consensus_ranking,
        final_results,
        feature_names,
    ):
        """Save analysis outputs to CSV files in the script directory."""
        print("\n6. Saving results...")

        selected_feature_set = set(final_results["selected_features"])
        rank_lookup = {
            feature_names[index]: rank
            for rank, index in enumerate(consensus_ranking, start=1)
        }
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "consensus_rank": [rank_lookup[name] for name in feature_names],
                "consensus_score": consensus_scores,
                "selected_in_best_model": [
                    name in selected_feature_set for name in feature_names
                ],
                "f_score": importance_results["f_score"]["scores"],
                "f_pvalue": importance_results["f_score"]["pvalues"],
                "mutual_info": importance_results["mutual_info"]["scores"],
                "random_forest": importance_results["random_forest"]["scores"],
                "svm_rfe_rank": importance_results["svm_rfe"]["ranking"],
                "correlation": importance_results["correlation"]["correlations"],
                "abs_correlation": importance_results["correlation"]["abs_correlations"],
            }
        ).sort_values("consensus_rank")
        importance_path = os.path.join(self.script_directory, "feature_importance_analysis.csv")
        importance_df.to_csv(importance_path, index=False)

        cv_results_path = os.path.join(
            self.script_directory,
            "feature_kernel_grid_search_results.csv",
        )
        final_results["cv_results"].sort_values("rank_test_score").to_csv(
            cv_results_path,
            index=False,
        )

        final_path = os.path.join(self.script_directory, "final_optimized_results.csv")
        pd.DataFrame(
            [
                {
                    "loso_accuracy": final_results["overall_accuracy"],
                    "best_cv_score": final_results["best_cv_score"],
                    "n_selected_features": len(final_results["selected_features"]),
                    "selected_features": ", ".join(final_results["selected_features"]),
                    "best_params": str(final_results["best_params"]),
                }
            ]
        ).to_csv(final_path, index=False)

        subject_path = os.path.join(self.script_directory, "subject_level_results.csv")
        pd.DataFrame(
            [
                {
                    "subject": subject,
                    "accuracy": result["accuracy"],
                    "n_samples": result["n_samples"],
                }
                for subject, result in sorted(final_results["subject_results"].items())
            ]
        ).to_csv(subject_path, index=False)

        print("Results saved:")
        print(f"  - {importance_path}")
        print(f"  - {cv_results_path}")
        print(f"  - {final_path}")
        print(f"  - {subject_path}")


def main():
    """Main execution function."""
    detector = FeatureImportanceHRVDetector()

    try:
        results = detector.run_complete_analysis()
        print("\nFEATURE IMPORTANCE ANALYSIS COMPLETE")
        print(f"Optimized LOSO Accuracy: {results['overall_accuracy']:.1%}")
        print(f"Best Features: {', '.join(results['selected_features'])}")
    except Exception as error:
        print(f"Error in analysis: {error}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()