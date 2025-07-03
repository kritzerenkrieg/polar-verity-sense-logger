#!/usr/bin/env python3
"""
HRV-Based Lie Detection with Feature Importance Analysis
First analyzes feature importance, then trains optimal SVM model
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceHRVDetector:
    def __init__(self, data_directory='.'):
        """
        Initialize the Feature Importance HRV Lie Detection system.
        
        Args:
            data_directory (str): Directory containing CSV files
        """
        self.data_directory = data_directory
        
        # All time and frequency domain features (your best performing model)
        self.time_domain_features = [
            'nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
            'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index'
        ]
        self.frequency_domain_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
            'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak'
        ]
        
        self.all_data = None
        self.feature_importance_results = {}
        
    def parse_filename(self, filename):
        """Parse CSV filename to extract metadata."""
        pattern = r'pr_([^_]+)_([^_]+)_(truth|lie)-sequence_(frequency_)?windowed\.csv'
        match = re.match(pattern, filename)
        
        if match:
            subject = match.group(1)
            condition = match.group(2)
            label = match.group(3)
            domain = 'frequency' if match.group(4) else 'time'
            
            return {
                'subject': subject,
                'condition': condition,
                'label': label,
                'domain': domain,
                'filename': filename
            }
        return None
    
    def discover_files(self):
        """Discover and organize all CSV files in the directory."""
        csv_files = [f for f in os.listdir(self.data_directory) 
                    if f.endswith('_windowed.csv') and not f.endswith('_validated_windowed.csv')]
        file_metadata = []
        
        for filename in csv_files:
            metadata = self.parse_filename(filename)
            if metadata:
                file_metadata.append(metadata)
                
        print(f"Discovered {len(file_metadata)} original windowed CSV files")
        
        subjects = set(meta['subject'] for meta in file_metadata)
        conditions = set(meta['condition'] for meta in file_metadata)
        
        print(f"Subjects found: {sorted(subjects)}")
        print(f"Conditions found: {sorted(conditions)}")
        
        return file_metadata
    
    def load_data(self, file_metadata):
        """Load and combine all CSV files into a single dataset."""
        all_dataframes = []
        
        for meta in file_metadata:
            filepath = os.path.join(self.data_directory, meta['filename'])
            
            try:
                df = pd.read_csv(filepath)
                
                # Add metadata columns
                df['subject'] = meta['subject']
                df['condition'] = meta['condition']
                df['label'] = meta['label']
                df['domain'] = meta['domain']
                df['binary_label'] = 1 if meta['label'] == 'lie' else 0
                
                all_dataframes.append(df)
                
            except Exception as e:
                print(f"Error loading {meta['filename']}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No valid CSV files could be loaded")
            
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} total samples")
        print(f"Label distribution: {combined_df['binary_label'].value_counts().to_dict()}")
        
        return combined_df
    
    def prepare_features(self, df):
        """
        Prepare feature matrix by merging time and frequency domains.
        
        Args:
            df (pd.DataFrame): Combined dataset
            
        Returns:
            tuple: (X, y, groups, feature_names)
        """
        # Merge time and frequency data
        time_df = df[df['domain'] == 'time'].copy()
        freq_df = df[df['domain'] == 'frequency'].copy()
        
        # Select columns for merging
        time_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.time_domain_features
        freq_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.frequency_domain_features
        
        # Only keep columns that exist
        time_cols = [col for col in time_cols if col in time_df.columns]
        freq_cols = [col for col in freq_cols if col in freq_df.columns]
        
        time_df_clean = time_df[time_cols].copy()
        freq_df_clean = freq_df[freq_cols].copy()
        
        # Merge on common identifiers
        merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
        merged_df = pd.merge(
            time_df_clean, freq_df_clean, 
            on=merge_cols, 
            suffixes=('_time', '_freq')
        )
        
        print(f"Merged time + frequency features: {len(merged_df)} samples")
        
        # Collect all feature columns
        all_features = []
        
        # Add time domain features
        for feature in self.time_domain_features:
            if feature in merged_df.columns:
                all_features.append(feature)
            elif f"{feature}_time" in merged_df.columns:
                all_features.append(f"{feature}_time")
        
        # Add frequency domain features
        for feature in self.frequency_domain_features:
            if feature in merged_df.columns:
                all_features.append(feature)
            elif f"{feature}_freq" in merged_df.columns:
                all_features.append(f"{feature}_freq")
        
        # Check for missing features
        available_features = [col for col in all_features if col in merged_df.columns]
        missing_features = [col for col in all_features if col not in merged_df.columns]
        
        if missing_features:
            print(f"Warning: Missing features {missing_features}")
        
        print(f"Using ALL {len(merged_df)} samples - no filtering")
        
        # Handle NaN values through imputation
        if merged_df[available_features].isnull().sum().sum() > 0:
            print("Imputing NaN values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            feature_data_imputed = imputer.fit_transform(merged_df[available_features])
            merged_df[available_features] = feature_data_imputed
        
        X = merged_df[available_features].values
        y = merged_df['binary_label'].values
        groups = merged_df['subject'].values
        
        print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Label distribution: Truth={np.sum(y==0)}, Lie={np.sum(y==1)}")
        print(f"Features: {available_features}")
        
        return X, y, groups, available_features
    
    def analyze_feature_importance(self, X, y, groups, feature_names):
        """
        Analyze feature importance using multiple methods.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            feature_names (list): Feature names
            
        Returns:
            dict: Feature importance results
        """
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        importance_results = {}
        
        # 1. Univariate Statistical Tests (F-score)
        print("\n1. Univariate F-score Analysis...")
        f_selector = SelectKBest(score_func=f_classif, k='all')
        f_selector.fit(X, y)
        f_scores = f_selector.scores_
        f_pvalues = f_selector.pvalues_
        
        importance_results['f_score'] = {
            'scores': f_scores,
            'pvalues': f_pvalues,
            'ranking': np.argsort(f_scores)[::-1]  # Descending order
        }
        
        print("Top 10 features by F-score:")
        for i, idx in enumerate(importance_results['f_score']['ranking'][:10]):
            print(f"  {i+1:2d}. {feature_names[idx]:<20} F={f_scores[idx]:.3f} p={f_pvalues[idx]:.4f}")
        
        # 2. Mutual Information
        print("\n2. Mutual Information Analysis...")
        mi_scores = mutual_info_classif(X, y, random_state=42)
        importance_results['mutual_info'] = {
            'scores': mi_scores,
            'ranking': np.argsort(mi_scores)[::-1]
        }
        
        print("Top 10 features by Mutual Information:")
        for i, idx in enumerate(importance_results['mutual_info']['ranking'][:10]):
            print(f"  {i+1:2d}. {feature_names[idx]:<20} MI={mi_scores[idx]:.4f}")
        
        # 3. Random Forest Feature Importance
        print("\n3. Random Forest Feature Importance...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        importance_results['random_forest'] = {
            'scores': rf_importance,
            'ranking': np.argsort(rf_importance)[::-1]
        }
        
        print("Top 10 features by Random Forest:")
        for i, idx in enumerate(importance_results['random_forest']['ranking'][:10]):
            print(f"  {i+1:2d}. {feature_names[idx]:<20} Importance={rf_importance[idx]:.4f}")
        
        # 4. SVM-RFE (Recursive Feature Elimination)
        print("\n4. SVM-RFE Analysis...")
        svm_estimator = SVC(kernel='linear', class_weight='balanced')
        rfe = RFE(estimator=svm_estimator, n_features_to_select=1, step=1)
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        rfe.fit(X_scaled, y)
        
        # RFE ranking (1 = best)
        rfe_ranking = rfe.ranking_
        importance_results['svm_rfe'] = {
            'ranking': rfe_ranking,
            'feature_ranking': np.argsort(rfe_ranking)  # Best features first
        }
        
        print("Top 10 features by SVM-RFE:")
        for i, idx in enumerate(importance_results['svm_rfe']['feature_ranking'][:10]):
            print(f"  {i+1:2d}. {feature_names[idx]:<20} Rank={rfe_ranking[idx]}")
        
        # 5. Correlation with Target
        print("\n5. Correlation Analysis...")
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        abs_correlations = np.abs(correlations)
        
        importance_results['correlation'] = {
            'correlations': correlations,
            'abs_correlations': abs_correlations,
            'ranking': np.argsort(abs_correlations)[::-1]
        }
        
        print("Top 10 features by Absolute Correlation:")
        for i, idx in enumerate(importance_results['correlation']['ranking'][:10]):
            print(f"  {i+1:2d}. {feature_names[idx]:<20} |r|={abs_correlations[idx]:.4f} r={correlations[idx]:.4f}")
        
        self.feature_importance_results = importance_results
        return importance_results
    
    def create_consensus_ranking(self, importance_results, feature_names):
        """
        Create consensus ranking from multiple importance methods.
        
        Args:
            importance_results (dict): Results from different importance methods
            feature_names (list): Feature names
            
        Returns:
            list: Consensus ranking of features
        """
        print(f"\n{'='*60}")
        print("CONSENSUS FEATURE RANKING")
        print(f"{'='*60}")
        
        n_features = len(feature_names)
        
        # Convert all rankings to normalized scores (0-1, higher = better)
        normalized_scores = {}
        
        # F-score (higher is better)
        f_scores = importance_results['f_score']['scores']
        normalized_scores['f_score'] = f_scores / (np.max(f_scores) + 1e-8)
        
        # Mutual Information (higher is better)
        mi_scores = importance_results['mutual_info']['scores']
        normalized_scores['mutual_info'] = mi_scores / (np.max(mi_scores) + 1e-8)
        
        # Random Forest (higher is better)
        rf_scores = importance_results['random_forest']['scores']
        normalized_scores['random_forest'] = rf_scores / (np.max(rf_scores) + 1e-8)
        
        # SVM-RFE (lower rank is better, convert to score)
        rfe_ranking = importance_results['svm_rfe']['ranking']
        normalized_scores['svm_rfe'] = (n_features - rfe_ranking + 1) / n_features
        
        # Correlation (higher absolute value is better)
        abs_corr = importance_results['correlation']['abs_correlations']
        normalized_scores['correlation'] = abs_corr / (np.max(abs_corr) + 1e-8)
        
        # Calculate consensus score (average of all methods)
        consensus_scores = np.zeros(n_features)
        for method_scores in normalized_scores.values():
            consensus_scores += method_scores
        consensus_scores /= len(normalized_scores)
        
        # Create consensus ranking
        consensus_ranking = np.argsort(consensus_scores)[::-1]
        
        print("CONSENSUS TOP 10 FEATURES:")
        print(f"{'Rank':<4} {'Feature':<20} {'Consensus':<10} {'F-score':<8} {'MI':<8} {'RF':<8} {'RFE':<8} {'Corr':<8}")
        print("-" * 80)
        
        for i, idx in enumerate(consensus_ranking[:10]):
            print(f"{i+1:<4} {feature_names[idx]:<20} {consensus_scores[idx]:.3f}    "
                  f"{normalized_scores['f_score'][idx]:.3f}   {normalized_scores['mutual_info'][idx]:.3f}   "
                  f"{normalized_scores['random_forest'][idx]:.3f}   {normalized_scores['svm_rfe'][idx]:.3f}   "
                  f"{normalized_scores['correlation'][idx]:.3f}")
        
        return consensus_ranking, consensus_scores
    
    def test_feature_subsets(self, X, y, groups, feature_names, consensus_ranking):
        """
        Test different feature subsets based on importance ranking.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels  
            groups (np.array): Subject groups
            feature_names (list): Feature names
            consensus_ranking (np.array): Consensus feature ranking
            
        Returns:
            dict: Results for different feature subset sizes
        """
        print(f"\n{'='*60}")
        print("TESTING FEATURE SUBSETS")
        print(f"{'='*60}")
        
        # Test different numbers of top features
        subset_sizes = [2, 3, 4, 5, 6, 8, 10, 12, 15, len(feature_names)]
        subset_results = {}
        
        logo = LeaveOneGroupOut()
        
        for n_features in subset_sizes:
            if n_features > len(feature_names):
                continue
                
            print(f"\nTesting top {n_features} features...")
            
            # Get top features
            top_features_idx = consensus_ranking[:n_features]
            top_features_names = [feature_names[i] for i in top_features_idx]
            X_subset = X[:, top_features_idx]
            
            print(f"Features: {top_features_names}")
            
            # Quick SVM test with optimized parameters
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=42))
            ])
            
            # LOSO CV
            scores = []
            for train_idx, test_idx in logo.split(X_subset, y, groups):
                X_train, X_test = X_subset[train_idx], X_subset[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_test, y_test)
                scores.append(score)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            subset_results[n_features] = {
                'features': top_features_names,
                'mean_score': mean_score,
                'std_score': std_score,
                'scores': scores
            }
            
            print(f"Accuracy: {mean_score:.4f} ± {std_score:.4f}")
        
        # Find best subset size
        best_size = max(subset_results.keys(), key=lambda k: subset_results[k]['mean_score'])
        best_score = subset_results[best_size]['mean_score']
        
        print(f"\n{'='*40}")
        print("FEATURE SUBSET RESULTS SUMMARY")
        print(f"{'='*40}")
        print(f"{'Features':<8} {'Accuracy':<12} {'Std':<8} {'Feature Names'}")
        print("-" * 70)
        
        for n_features in sorted(subset_results.keys()):
            result = subset_results[n_features]
            marker = " ⭐" if n_features == best_size else ""
            print(f"{n_features:<8} {result['mean_score']:<12.4f} {result['std_score']:<8.4f} "
                  f"{', '.join(result['features'][:3])}{'...' if len(result['features']) > 3 else ''}{marker}")
        
        print(f"\nBest performance: {best_score:.4f} with {best_size} features")
        
        return subset_results, best_size
    
    def train_final_optimized_model(self, X, y, groups, feature_names, best_features):
        """
        Train final optimized model with best features and full hyperparameter tuning.
        
        Args:
            X (np.array): Feature matrix (subset to best features)
            y (np.array): Labels
            groups (np.array): Subject groups
            feature_names (list): Best feature names
            best_features (list): Indices of best features
            
        Returns:
            dict: Final model results
        """
        print(f"\n{'='*60}")
        print("FINAL OPTIMIZED MODEL TRAINING")
        print(f"{'='*60}")
        
        X_best = X[:, best_features]
        
        print(f"Training final model with {len(best_features)} best features:")
        for i, idx in enumerate(best_features):
            print(f"  {i+1}. {feature_names[idx]}")
        
        # Comprehensive hyperparameter tuning for final model
        param_grids = {
            'linear': {
                'svm__C': [0.1, 1, 10, 100]
            },
            'rbf': {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
            },
            'poly': {
                'svm__C': [0.1, 1, 10],
                'svm__gamma': ['scale', 'auto'],
                'svm__degree': [2, 3]
            }
        }
        
        logo = LeaveOneGroupOut()
        kernel_results = {}
        
        for kernel, param_grid in param_grids.items():
            print(f"\nOptimizing {kernel} kernel...")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, random_state=42, class_weight='balanced'))
            ])
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=logo, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_best, y, groups=groups)
            
            kernel_results[kernel] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            print(f"{kernel}: {grid_search.best_score_:.4f} with {grid_search.best_params_}")
        
        # Select best kernel
        best_kernel = max(kernel_results.keys(), key=lambda k: kernel_results[k]['best_score'])
        best_params = kernel_results[best_kernel]['best_params']
        best_cv_score = kernel_results[best_kernel]['best_score']
        
        print(f"\nBest kernel: {best_kernel} with CV score: {best_cv_score:.4f}")
        
        # Final evaluation
        svm_params = {k.replace('svm__', ''): v for k, v in best_params.items()}
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**svm_params, kernel=best_kernel, random_state=42, class_weight='balanced'))
        ])
        
        # Detailed LOSO evaluation
        y_true_all = []
        y_pred_all = []
        subject_results = {}
        
        for train_idx, test_idx in logo.split(X_best, y, groups):
            X_train, X_test = X_best[train_idx], X_best[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = groups[test_idx][0]
            
            final_pipeline.fit(X_train, y_train)
            y_pred = final_pipeline.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            subject_results[test_subject] = {
                'accuracy': accuracy,
                'n_samples': len(y_test)
            }
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            print(f"Subject {test_subject}: {accuracy:.4f} ({len(y_test)} samples)")
        
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        
        results = {
            'overall_accuracy': overall_accuracy,
            'best_features': [feature_names[i] for i in best_features],
            'best_kernel': best_kernel,
            'best_params': best_params,
            'subject_results': subject_results,
            'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
            'classification_report': classification_report(y_true_all, y_pred_all, target_names=['Truth', 'Lie'])
        }
        
        return results
    
    def run_complete_analysis(self):
        """
        Run complete feature importance analysis and model optimization.
        """
        print("=== HRV LIE DETECTION WITH FEATURE IMPORTANCE ANALYSIS ===")
        
        # Step 1: Load data
        print("\n1. Loading data...")
        file_metadata = self.discover_files()
        self.all_data = self.load_data(file_metadata)
        
        # Step 2: Prepare features  
        print("\n2. Preparing features...")
        X, y, groups, feature_names = self.prepare_features(self.all_data)
        
        # Step 3: Analyze feature importance
        print("\n3. Analyzing feature importance...")
        importance_results = self.analyze_feature_importance(X, y, groups, feature_names)
        
        # Step 4: Create consensus ranking
        print("\n4. Creating consensus ranking...")
        consensus_ranking, consensus_scores = self.create_consensus_ranking(importance_results, feature_names)
        
        # Step 5: Test feature subsets
        print("\n5. Testing feature subsets...")
        subset_results, best_size = self.test_feature_subsets(X, y, groups, feature_names, consensus_ranking)
        
        # Step 6: Train final optimized model
        print("\n6. Training final optimized model...")
        best_features = consensus_ranking[:best_size]
        final_results = self.train_final_optimized_model(X, y, groups, feature_names, best_features)
        
        # Step 7: Display final results
        print(f"\n{'='*80}")
        print("FINAL OPTIMIZED RESULTS")
        print(f"{'='*80}")
        print(f"Best Feature Subset Size: {best_size}")
        print(f"Selected Features: {final_results['best_features']}")
        print(f"Final Accuracy: {final_results['overall_accuracy']:.4f}")
        print(f"Best Kernel: {final_results['best_kernel']}")
        print(f"\nConfusion Matrix:")
        print(final_results['confusion_matrix'])
        
        # Save results
        self.save_importance_results(importance_results, subset_results, final_results, feature_names)
        
        return final_results
    
    def save_importance_results(self, importance_results, subset_results, final_results, feature_names):
        """Save all analysis results to CSV files."""
        print(f"\n7. Saving results...")
        
        # 1. Feature importance rankings
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'f_score': importance_results['f_score']['scores'],
            'mutual_info': importance_results['mutual_info']['scores'],
            'random_forest': importance_results['random_forest']['scores'],
            'svm_rfe_rank': importance_results['svm_rfe']['ranking'],
            'correlation': importance_results['correlation']['correlations'],
            'abs_correlation': importance_results['correlation']['abs_correlations']
        })
        importance_df.to_csv('feature_importance_analysis.csv', index=False)
        
        # 2. Subset testing results
        subset_df = pd.DataFrame([
            {
                'n_features': n,
                'accuracy': results['mean_score'],
                'std': results['std_score'],
                'features': ', '.join(results['features'])
            }
            for n, results in subset_results.items()
        ])
        subset_df.to_csv('feature_subset_results.csv', index=False)
        
        # 3. Final model results
        final_df = pd.DataFrame([{
            'accuracy': final_results['overall_accuracy'],
            'n_features': len(final_results['best_features']),
            'features': ', '.join(final_results['best_features']),
            'kernel': final_results['best_kernel'],
            'params': str(final_results['best_params'])
        }])
        final_df.to_csv('final_optimized_results.csv', index=False)
        
        print("Results saved:")
        print("  - feature_importance_analysis.csv")
        print("  - feature_subset_results.csv") 
        print("  - final_optimized_results.csv")

def main():
    """Main execution function."""
    detector = FeatureImportanceHRVDetector(data_directory='.')
    
    try:
        results = detector.run_complete_analysis()
        
        accuracy = results['overall_accuracy']
        n_features = len(results['best_features'])
        
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS COMPLETE!")
        print(f"📊 Optimized Accuracy: {accuracy:.1%}")
        print(f"🔧 Best Features ({n_features}): {', '.join(results['best_features'])}")
        
        if accuracy > 0.56:
            print(f"🚀 SUCCESS! Beat your best model of 56%!")
        elif accuracy > 0.55:
            print(f"✅ CLOSE! Very close to your best 56%")
        else:
            print(f"📋 Analysis shows optimal feature combination")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()