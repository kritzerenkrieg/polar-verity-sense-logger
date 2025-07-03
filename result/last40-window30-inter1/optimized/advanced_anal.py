#!/usr/bin/env python3
"""
Advanced HRV Feature Optimization
Improves upon initial combinatorial results using:
1. Correlation analysis and collinearity removal
2. Recursive feature elimination (RFE)
3. Feature importance analysis
4. Advanced kernel optimization
5. Ensemble-based feature selection
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class AdvancedHRVOptimizer:
    def __init__(self, data_directory='.'):
        """
        Initialize the Advanced HRV Optimization system.
        
        Args:
            data_directory (str): Directory containing CSV files
        """
        self.data_directory = data_directory
        
        # Top performing combinations from initial analysis
        self.top_combinations = [
            # Top 5 from your results
            {
                'name': 'best_12feat',
                'features': ['nn_count', 'nn_mean', 'nn_min', 'sdnn', 'sdsd', 'pnn20', 
                           'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 'hf_norm', 'lf_peak'],
                'baseline_accuracy': 0.6875
            },
            {
                'name': 'best_8feat',
                'features': ['nn_count', 'nn_mean', 'nn_min', 'sdsd', 'rmssd', 'pnn20', 
                           'lf_power', 'hf_peak'],
                'baseline_accuracy': 0.6864
            },
            {
                'name': 'best_4feat',
                'features': ['nn_min', 'pnn20', 'lf_power', 'hf_norm'],
                'baseline_accuracy': 0.6818
            },
            {
                'name': 'best_5feat',
                'features': ['nn_count', 'sdnn', 'pnn20', 'lf_power', 'lf_norm'],
                'baseline_accuracy': 0.6795
            }
        ]
        
        # Simplified kernel configurations for faster training
        self.advanced_kernel_configs = {
            'linear': {
                'svm__C': [0.1, 0.5, 1, 2, 5, 10, 50, 100]
            },
            'rbf': {
                'svm__C': [0.1, 0.5, 1, 2, 5, 10, 50, 100],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1, 0.5, 1]
            },
            # Removed poly and sigmoid for speed
        }
        
        self.all_data = None
        self.optimization_results = []
        
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
    
    def load_data(self):
        """Load and combine all CSV files into a single dataset."""
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        file_metadata = []
        
        for filename in csv_files:
            metadata = self.parse_filename(filename)
            if metadata:
                file_metadata.append(metadata)
        
        all_dataframes = []
        
        for meta in file_metadata:
            filepath = os.path.join(self.data_directory, meta['filename'])
            
            try:
                df = pd.read_csv(filepath)
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
        
        return combined_df
    
    def prepare_features_for_combination(self, df, feature_list):
        """Prepare feature matrix for a specific feature combination."""
        # Separate time and frequency data
        time_df = df[df['domain'] == 'time'].copy()
        freq_df = df[df['domain'] == 'frequency'].copy()
        
        # Define time and frequency domain features
        time_domain_features = [
            'nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
            'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index'
        ]
        frequency_domain_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
            'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak'
        ]
        
        # Merge columns
        merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
        
        # Separate features by domain
        time_features_needed = [f for f in feature_list if f in time_domain_features]
        freq_features_needed = [f for f in feature_list if f in frequency_domain_features]
        
        if len(time_features_needed) > 0 and len(freq_features_needed) > 0:
            # Need both domains
            time_cols = merge_cols + time_features_needed
            freq_cols = merge_cols + freq_features_needed
            
            time_df_clean = time_df[[col for col in time_cols if col in time_df.columns]].copy()
            freq_df_clean = freq_df[[col for col in freq_cols if col in freq_df.columns]].copy()
            
            df_merged = pd.merge(time_df_clean, freq_df_clean, on=merge_cols, suffixes=('_time', '_freq'))
            
            # Get available features
            available_features = []
            for feature in feature_list:
                if feature in time_features_needed and feature in df_merged.columns:
                    available_features.append(feature)
                elif feature in freq_features_needed and feature in df_merged.columns:
                    available_features.append(feature)
                    
        elif len(time_features_needed) > 0:
            available_features = [f for f in time_features_needed if f in time_df.columns]
            df_merged = time_df[merge_cols + available_features].copy()
        else:
            available_features = [f for f in freq_features_needed if f in freq_df.columns]
            df_merged = freq_df[merge_cols + available_features].copy()
        
        # Clean data
        df_clean = df_merged.dropna(subset=available_features)
        
        if len(df_clean) == 0 or len(available_features) == 0:
            return None, None, None, []
        
        X = df_clean[available_features].values
        y = df_clean['binary_label'].values
        groups = df_clean['subject'].values
        
        return X, y, groups, available_features
    
    def analyze_feature_correlations(self, X, feature_names, threshold=0.9):
        """
        Analyze feature correlations and identify highly correlated pairs.
        
        Args:
            X (np.array): Feature matrix
            feature_names (list): List of feature names
            threshold (float): Correlation threshold for removal
            
        Returns:
            dict: Analysis results including correlations and recommendations
        """
        print(f"\n--- Correlation Analysis ---")
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        features_to_remove = set()
        
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_value = abs(corr_matrix[i, j])
                if corr_value > threshold:
                    high_corr_pairs.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr_value
                    })
                    # Remove the feature that appears later in the list
                    features_to_remove.add(feature_names[j])
        
        print(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > {threshold})")
        for pair in high_corr_pairs:
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: r = {pair['correlation']:.3f}")
        
        if features_to_remove:
            print(f"Recommended for removal: {list(features_to_remove)}")
        else:
            print("No highly correlated features found")
        
        return {
            'correlation_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'features_to_remove': list(features_to_remove),
            'cleaned_features': [f for f in feature_names if f not in features_to_remove]
        }
    
    def recursive_feature_elimination(self, X, y, groups, feature_names, target_features=None):
        """
        Use Recursive Feature Elimination to find optimal feature subset.
        Fixed to use linear SVM which has coef_ attribute for importance.
        
        Args:
            X, y, groups: Data arrays
            feature_names (list): Feature names
            target_features (int): Target number of features (if None, test multiple)
            
        Returns:
            dict: RFE results with optimal features
        """
        print(f"\n--- Recursive Feature Elimination ---")
        
        if target_features is None:
            # Test different numbers of features
            feature_counts = [3, 4, 5, 6, 7, 8, min(10, len(feature_names))]
        else:
            feature_counts = [target_features]
        
        logo = LeaveOneGroupOut()
        best_score = 0
        best_features = None
        best_count = 0
        
        results = []
        
        for n_features in feature_counts:
            if n_features >= len(feature_names):
                continue
                
            print(f"Testing {n_features} features...")
            
            # Use Linear SVM as the estimator for RFE (has coef_ attribute)
            svm_estimator = SVC(kernel='linear', C=1.0)
            
            # Perform RFE
            rfe = RFE(estimator=svm_estimator, n_features_to_select=n_features)
            
            # Evaluate using cross-validation
            scores = []
            selected_features_list = []
            
            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Fit RFE and transform data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                rfe.fit(X_train_scaled, y_train)
                X_train_rfe = rfe.transform(X_train_scaled)
                X_test_rfe = rfe.transform(X_test_scaled)
                
                # Train SVM on selected features (use RBF for final evaluation)
                svm = SVC(kernel='rbf', C=1.0, gamma='scale')
                svm.fit(X_train_rfe, y_train)
                y_pred = svm.predict(X_test_rfe)
                
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)
                
                # Track selected features
                selected_features = [feature_names[i] for i, selected in enumerate(rfe.support_) if selected]
                selected_features_list.append(selected_features)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Find most commonly selected features across folds
            feature_counts_dict = {}
            for fold_features in selected_features_list:
                for feature in fold_features:
                    feature_counts_dict[feature] = feature_counts_dict.get(feature, 0) + 1
            
            # Select features that appear in most folds
            common_features = sorted(feature_counts_dict.items(), key=lambda x: x[1], reverse=True)[:n_features]
            stable_features = [feature for feature, count in common_features]
            
            result = {
                'n_features': n_features,
                'avg_score': avg_score,
                'std_score': std_score,
                'selected_features': stable_features,
                'feature_stability': {feature: count/len(selected_features_list) for feature, count in common_features}
            }
            results.append(result)
            
            print(f"  {n_features} features: {avg_score:.4f} ± {std_score:.4f}")
            print(f"  Selected: {stable_features}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_features = stable_features
                best_count = n_features
        
        print(f"\nBest RFE result: {best_count} features, accuracy = {best_score:.4f}")
        print(f"Best features: {best_features}")
        
    def variance_threshold_selection(self, X, y, groups, feature_names):
        """
        Alternative feature selection using variance threshold and univariate selection.
        More robust than RFE for SVM.
        
        Args:
            X, y, groups: Data arrays
            feature_names (list): Feature names
            
        Returns:
            dict: Feature selection results
        """
        print(f"\n--- Variance Threshold & Univariate Feature Selection ---")
        
        from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
        
        logo = LeaveOneGroupOut()
        
        # Method 1: Remove low variance features
        variance_threshold = VarianceThreshold(threshold=0.01)  # Remove features with very low variance
        
        # Method 2: Univariate feature selection
        results = []
        
        for k in [3, 4, 5, 6, 7, 8, min(10, len(feature_names))]:
            if k >= len(feature_names):
                continue
                
            print(f"Testing top {k} features by F-score...")
            
            scores = []
            selected_features_list = []
            
            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Apply variance threshold
                var_selector = VarianceThreshold(threshold=0.01)
                X_train_var = var_selector.fit_transform(X_train_scaled)
                X_test_var = var_selector.transform(X_test_scaled)
                
                # Get feature names after variance filtering
                var_feature_mask = var_selector.get_support()
                var_features = [feature_names[i] for i, selected in enumerate(var_feature_mask) if selected]
                
                # Apply univariate selection on remaining features
                if len(var_features) > k:
                    selector = SelectKBest(score_func=f_classif, k=k)
                    X_train_selected = selector.fit_transform(X_train_var, y_train)
                    X_test_selected = selector.transform(X_test_var)
                    
                    # Get final selected features
                    selected_mask = selector.get_support()
                    selected_features = [var_features[i] for i, selected in enumerate(selected_mask) if selected]
                else:
                    X_train_selected = X_train_var
                    X_test_selected = X_test_var
                    selected_features = var_features
                
                selected_features_list.append(selected_features)
                
                # Train SVM on selected features
                svm = SVC(kernel='rbf', C=1.0, gamma='scale')
                svm.fit(X_train_selected, y_train)
                y_pred = svm.predict(X_test_selected)
                
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)
            
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Find most commonly selected features
            feature_counts = {}
            for fold_features in selected_features_list:
                for feature in fold_features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            # Get most stable features
            stable_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:k]
            final_features = [feature for feature, count in stable_features]
            
            result = {
                'k': k,
                'avg_score': avg_score,
                'std_score': std_score,
                'selected_features': final_features,
                'feature_stability': {feature: count/len(selected_features_list) for feature, count in stable_features}
            }
            results.append(result)
            
            print(f"  Top {k} features: {avg_score:.4f} ± {std_score:.4f}")
            print(f"  Selected: {final_features}")
        
        # Find best result
        best_result = max(results, key=lambda x: x['avg_score'])
        
        print(f"\nBest univariate selection: {best_result['k']} features, accuracy = {best_result['avg_score']:.4f}")
        print(f"Best features: {best_result['selected_features']}")
        
    def forward_selection_svm(self, X, y, groups, feature_names, max_features=8):
        """
        Forward feature selection specifically designed for SVM.
        Adds features one by one based on cross-validation performance.
        
        Args:
            X, y, groups: Data arrays
            feature_names (list): Feature names
            max_features (int): Maximum number of features to select
            
        Returns:
            dict: Forward selection results
        """
        print(f"\n--- Forward Feature Selection for SVM ---")
        
        logo = LeaveOneGroupOut()
        
        selected_features = []
        remaining_features = feature_names.copy()
        selection_history = []
        
        # Start with the best single feature
        best_single_score = 0
        best_single_feature = None
        
        print("Finding best starting feature...")
        for feature in feature_names:
            feature_idx = feature_names.index(feature)
            X_single = X[:, [feature_idx]]
            
            scores = []
            for train_idx, test_idx in logo.split(X_single, y, groups):
                X_train, X_test = X_single[train_idx], X_single[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                svm = SVC(kernel='rbf', C=1.0, gamma='scale')
                svm.fit(X_train_scaled, y_train)
                y_pred = svm.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)
            
            avg_score = np.mean(scores)
            if avg_score > best_single_score:
                best_single_score = avg_score
                best_single_feature = feature
        
        selected_features.append(best_single_feature)
        remaining_features.remove(best_single_feature)
        
        print(f"Best starting feature: {best_single_feature} (score: {best_single_score:.4f})")
        
        selection_history.append({
            'step': 1,
            'added_feature': best_single_feature,
            'selected_features': selected_features.copy(),
            'score': best_single_score
        })
        
        # Forward selection loop
        for step in range(2, max_features + 1):
            if not remaining_features:
                break
                
            print(f"\nStep {step}: Adding feature {step}...")
            
            best_score = 0
            best_feature = None
            
            for candidate_feature in remaining_features:
                # Test adding this feature
                test_features = selected_features + [candidate_feature]
                test_indices = [feature_names.index(f) for f in test_features]
                X_test = X[:, test_indices]
                
                scores = []
                for train_idx, test_idx in logo.split(X_test, y, groups):
                    X_train, X_test_fold = X_test[train_idx], X_test[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test_fold)
                    
                    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
                    svm.fit(X_train_scaled, y_train)
                    y_pred = svm.predict(X_test_scaled)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    scores.append(accuracy)
                
                avg_score = np.mean(scores)
                print(f"  Testing {candidate_feature}: {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_feature = candidate_feature
            
            if best_feature is None:
                print(f"No improvement found, stopping at {len(selected_features)} features")
                break
            
            # Add the best feature
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            
            print(f"  Added: {best_feature} (score: {best_score:.4f})")
            print(f"  Current features: {selected_features}")
            
            selection_history.append({
                'step': step,
                'added_feature': best_feature,
                'selected_features': selected_features.copy(),
                'score': best_score
            })
            
            # Early stopping if no significant improvement
            if len(selection_history) >= 2:
                improvement = best_score - selection_history[-2]['score']
                if improvement < 0.005:  # Less than 0.5% improvement
                    print(f"Small improvement ({improvement:.4f}), considering early stop...")
        
        # Find best combination from history
        best_combination = max(selection_history, key=lambda x: x['score'])
        
        print(f"\nBest forward selection: {len(best_combination['selected_features'])} features")
        print(f"Score: {best_combination['score']:.4f}")
        print(f"Features: {best_combination['selected_features']}")
        
        return {
            'best_score': best_combination['score'],
            'best_features': best_combination['selected_features'],
            'best_count': len(best_combination['selected_features']),
            'selection_history': selection_history,
            'all_results': selection_history
        }
    
    def feature_importance_analysis(self, X, y, groups, feature_names):
        """
        Analyze feature importance using multiple methods.
        
        Args:
            X, y, groups: Data arrays
            feature_names (list): Feature names
            
        Returns:
            dict: Feature importance results
        """
        print(f"\n--- Feature Importance Analysis ---")
        
        logo = LeaveOneGroupOut()
        
        # Method 1: Random Forest feature importance
        rf_importances = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, y_train = X[train_idx], y[train_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_importances.append(rf.feature_importances_)
        
        avg_rf_importance = np.mean(rf_importances, axis=0)
        
        # Method 2: Univariate feature selection (F-score)
        f_scores, _ = f_classif(StandardScaler().fit_transform(X), y)
        
        # Method 3: Mutual information
        mi_scores = mutual_info_classif(StandardScaler().fit_transform(X), y, random_state=42)
        
        # Combine results
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'rf_importance': avg_rf_importance,
            'f_score': f_scores,
            'mutual_info': mi_scores
        })
        
        # Normalize scores to 0-1 range
        for col in ['rf_importance', 'f_score', 'mutual_info']:
            importance_df[f'{col}_norm'] = (importance_df[col] - importance_df[col].min()) / (importance_df[col].max() - importance_df[col].min())
        
        # Calculate combined score
        importance_df['combined_score'] = (
            importance_df['rf_importance_norm'] + 
            importance_df['f_score_norm'] + 
            importance_df['mutual_info_norm']
        ) / 3
        
        # Sort by combined score
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        
        print("Feature importance ranking:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:<15} | Combined: {row['combined_score']:.3f} | RF: {row['rf_importance']:.3f} | F: {row['f_score']:.1f} | MI: {row['mutual_info']:.3f}")
        
        return importance_df
    
    def advanced_hyperparameter_optimization(self, X, y, groups, feature_names):
        """
        Perform advanced hyperparameter optimization with extended search space.
        
        Args:
            X, y, groups: Data arrays
            feature_names (list): Feature names
            
        Returns:
            dict: Best hyperparameters and performance
        """
        print(f"\n--- Advanced Hyperparameter Optimization ---")
        
        logo = LeaveOneGroupOut()
        best_score = 0
        best_config = None
        
        for kernel, param_grid in self.advanced_kernel_configs.items():
            print(f"\nOptimizing {kernel} kernel...")
            
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, random_state=42))
            ])
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=logo,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            try:
                grid_search.fit(X, y, groups=groups)
                
                print(f"  Best {kernel} score: {grid_search.best_score_:.4f}")
                print(f"  Best {kernel} params: {grid_search.best_params_}")
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_config = {
                        'kernel': kernel,
                        'params': grid_search.best_params_,
                        'score': grid_search.best_score_
                    }
                    
            except Exception as e:
                print(f"  Error with {kernel} kernel: {e}")
                continue
        
        # Fallback if no kernel worked
        if best_config is None:
            print(f"  All kernels failed, using default RBF")
            best_config = {
                'kernel': 'rbf',
                'params': {'svm__C': 1.0, 'svm__gamma': 'scale'},
                'score': 0.5  # Default score
            }
        
        print(f"\nBest configuration:")
        print(f"  Kernel: {best_config['kernel']}")
        print(f"  Score: {best_config['score']:.4f}")
        print(f"  Params: {best_config['params']}")
        
        return best_config
    
    def optimize_combination(self, combination_info):
        """
        Apply all optimization techniques to a single feature combination.
        
        Args:
            combination_info (dict): Combination with name, features, and baseline accuracy
            
        Returns:
            dict: Optimization results
        """
        name = combination_info['name']
        features = combination_info['features']
        baseline_accuracy = combination_info['baseline_accuracy']
        
        print(f"\n{'='*80}")
        print(f"OPTIMIZING: {name}")
        print(f"Original features ({len(features)}): {features}")
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        print("="*80)
        
        # Prepare data
        X, y, groups, available_features = self.prepare_features_for_combination(self.all_data, features)
        
        if X is None:
            print(f"❌ Could not prepare features for {name}")
            return {'name': name, 'status': 'failed'}
        
        print(f"Data prepared: {X.shape[0]} samples, {len(available_features)} features")
        
        results = {'name': name, 'baseline_accuracy': baseline_accuracy, 'optimizations': {}}
        
        # 1. Correlation analysis
        corr_analysis = self.analyze_feature_correlations(X, available_features, threshold=0.8)
        results['optimizations']['correlation'] = corr_analysis
        
        # 2. Feature importance analysis
        importance_analysis = self.feature_importance_analysis(X, y, groups, available_features)
        results['optimizations']['importance'] = importance_analysis
        
        # 3. Feature selection (try multiple methods)
        feature_selection_results = {}
        
        # Method 1: Try RFE first
        try:
            print(f"\nTrying Recursive Feature Elimination...")
            rfe_analysis = self.recursive_feature_elimination(X, y, groups, available_features)
            feature_selection_results['rfe'] = rfe_analysis
        except Exception as e:
            print(f"RFE failed: {e}")
        
        # Method 2: Univariate selection (always works)
        try:
            print(f"\nTrying Univariate Feature Selection...")
            univariate_analysis = self.variance_threshold_selection(X, y, groups, available_features)
            feature_selection_results['univariate'] = univariate_analysis
        except Exception as e:
            print(f"Univariate selection failed: {e}")
        
        # Method 3: Forward selection (most robust for SVM)
        try:
            print(f"\nTrying Forward Feature Selection...")
            forward_analysis = self.forward_selection_svm(X, y, groups, available_features, max_features=8)
            feature_selection_results['forward'] = forward_analysis
        except Exception as e:
            print(f"Forward selection failed: {e}")
        
        # Choose best feature selection method
        valid_selection_results = {k: v for k, v in feature_selection_results.items() if v is not None and 'best_score' in v}
        
        if valid_selection_results:
            best_method = max(valid_selection_results.keys(), 
                            key=lambda k: valid_selection_results[k]['best_score'])
            best_selection = valid_selection_results[best_method]
            print(f"\nBest feature selection method: {best_method}")
            print(f"Score: {best_selection['best_score']:.4f}")
            print(f"Features: {best_selection['best_features']}")
        else:
            # Fallback: use all features
            best_selection = {
                'best_score': 0,
                'best_features': available_features,
                'best_count': len(available_features)
            }
            print(f"\nAll feature selection methods failed, using all features")
        
        results['optimizations']['feature_selection'] = feature_selection_results
        results['optimizations']['best_selection'] = best_selection
        
        # 5. Test optimized feature sets
        optimization_tests = []
        
        # Test 5a: Correlation-cleaned features
        if corr_analysis['cleaned_features']:
            print(f"\n--- Testing correlation-cleaned features ---")
            try:
                cleaned_X, cleaned_y, cleaned_groups, cleaned_available = self.prepare_features_for_combination(
                    self.all_data, corr_analysis['cleaned_features']
                )
                if cleaned_X is not None:
                    best_config = self.advanced_hyperparameter_optimization(cleaned_X, cleaned_y, cleaned_groups, cleaned_available)
                    optimization_tests.append({
                        'method': 'correlation_cleaned',
                        'features': cleaned_available,
                        'n_features': len(cleaned_available),
                        'accuracy': best_config['score'],
                        'config': best_config,
                        'improvement': best_config['score'] - baseline_accuracy
                    })
            except Exception as e:
                print(f"Error testing correlation-cleaned features: {e}")
        
        # Test 5b: Best feature selection method
        if best_selection['best_features']:
            print(f"\n--- Testing feature selection results ---")
            try:
                selection_X, selection_y, selection_groups, selection_available = self.prepare_features_for_combination(
                    self.all_data, best_selection['best_features']
                )
                if selection_X is not None:
                    best_config = self.advanced_hyperparameter_optimization(selection_X, selection_y, selection_groups, selection_available)
                    optimization_tests.append({
                        'method': 'feature_selection',
                        'features': selection_available,
                        'n_features': len(selection_available),
                        'accuracy': best_config['score'],
                        'config': best_config,
                        'improvement': best_config['score'] - baseline_accuracy
                    })
            except Exception as e:
                print(f"Error testing feature selection results: {e}")
        
        # Test 5c: Top importance features (top 50% by combined score)
        try:
            top_features = importance_analysis.head(max(3, len(importance_analysis)//2))['feature'].tolist()
            print(f"\n--- Testing top importance features ---")
            imp_X, imp_y, imp_groups, imp_available = self.prepare_features_for_combination(
                self.all_data, top_features
            )
            if imp_X is not None:
                best_config = self.advanced_hyperparameter_optimization(imp_X, imp_y, imp_groups, imp_available)
                optimization_tests.append({
                    'method': 'importance_selected',
                    'features': imp_available,
                    'n_features': len(imp_available),
                    'accuracy': best_config['score'],
                    'config': best_config,
                    'improvement': best_config['score'] - baseline_accuracy
                })
        except Exception as e:
            print(f"Error testing importance features: {e}")
        
        # Test 5d: Original features with advanced hyperparameter optimization
        try:
            print(f"\n--- Testing original features with advanced hyperparameters ---")
            best_config = self.advanced_hyperparameter_optimization(X, y, groups, available_features)
            optimization_tests.append({
                'method': 'advanced_hyperparams',
                'features': available_features,
                'n_features': len(available_features),
                'accuracy': best_config['score'],
                'config': best_config,
                'improvement': best_config['score'] - baseline_accuracy
            })
        except Exception as e:
            print(f"Error testing advanced hyperparameters: {e}") == self.advanced_hyperparameter_optimization(imp_X, imp_y, imp_groups, imp_available)
            optimization_tests.append({
                'method': 'importance_selected',
                'features': imp_available,
                'n_features': len(imp_available),
                'accuracy': best_config['score'],
                'config': best_config,
                'improvement': best_config['score'] - baseline_accuracy
            })
        
        # Test 4d: Original features with advanced hyperparameter optimization
        print(f"\n--- Testing original features with advanced hyperparameters ---")
        best_config = self.advanced_hyperparameter_optimization(X, y, groups, available_features)
        optimization_tests.append({
            'method': 'advanced_hyperparams',
            'features': available_features,
            'n_features': len(available_features),
            'accuracy': best_config['score'],
            'config': best_config,
            'improvement': best_config['score'] - baseline_accuracy
        })
        
        results['optimizations']['tests'] = optimization_tests
        
        # Find best optimization
        best_optimization = max(optimization_tests, key=lambda x: x['accuracy'])
        results['best_optimization'] = best_optimization
        
        print(f"\n🎯 OPTIMIZATION SUMMARY for {name}")
        print(f"Baseline accuracy: {baseline_accuracy:.4f}")
        print(f"Best optimized accuracy: {best_optimization['accuracy']:.4f}")
        print(f"Improvement: +{best_optimization['improvement']*100:.2f}%")
        print(f"Best method: {best_optimization['method']}")
        print(f"Best features ({best_optimization['n_features']}): {best_optimization['features']}")
        print(f"Best config: {best_optimization['config']['kernel']} with {best_optimization['config']['params']}")
        
        return results
    
    def run_advanced_optimization(self):
        """Run advanced optimization on top performing combinations."""
        print("=== Advanced HRV Feature Optimization ===")
        print("Applying advanced techniques to improve top combinations")
        
        # Load data
        print("\n1. Loading data...")
        self.all_data = self.load_data()
        
        # Optimize each top combination
        print(f"\n2. Optimizing {len(self.top_combinations)} top combinations...")
        
        optimization_results = []
        
        for i, combination in enumerate(self.top_combinations):
            print(f"\n[{i+1}/{len(self.top_combinations)}]")
            result = self.optimize_combination(combination)
            optimization_results.append(result)
        
        # Summary of all optimizations
        print(f"\n\n{'='*80}")
        print("FINAL OPTIMIZATION SUMMARY")
        print("="*80)
        
        successful_results = [r for r in optimization_results if 'best_optimization' in r]
        successful_results.sort(key=lambda x: x['best_optimization']['accuracy'], reverse=True)
        
        print(f"\nRANKING OF OPTIMIZED COMBINATIONS:")
        print("-"*80)
        print(f"{'Rank':<4} {'Name':<15} {'Method':<20} {'Baseline':<8} {'Optimized':<9} {'Improvement':<12} {'#Feat'}")
        print("-"*80)
        
        for i, result in enumerate(successful_results):
            best_opt = result['best_optimization']
            improvement_pct = best_opt['improvement'] * 100
            
            print(f"{i+1:<4} {result['name']:<15} {best_opt['method']:<20} {result['baseline_accuracy']:<8.4f} {best_opt['accuracy']:<9.4f} {improvement_pct:+6.2f}%      {best_opt['n_features']}")
        
        # Save results
        self.save_advanced_results(optimization_results)
        
        return optimization_results
    
    def save_advanced_results(self, results):
        """Save advanced optimization results."""
        print(f"\n3. Saving results...")
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            if 'best_optimization' in result:
                best_opt = result['best_optimization']
                summary_data.append({
                    'combination_name': result['name'],
                    'baseline_accuracy': result['baseline_accuracy'],
                    'optimized_accuracy': best_opt['accuracy'],
                    'improvement_percent': best_opt['improvement'] * 100,
                    'best_method': best_opt['method'],
                    'n_features': best_opt['n_features'],
                    'features': ', '.join(best_opt['features']),
                    'kernel': best_opt['config']['kernel'],
                    'kernel_params': str(best_opt['config']['params'])
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('optimized_accuracy', ascending=False)
        summary_df.to_csv('advanced_optimization_results.csv', index=False)
        
        print("✅ Results saved to 'advanced_optimization_results.csv'")

def main():
    """Main execution function."""
    optimizer = AdvancedHRVOptimizer(data_directory='.')
    
    print("Advanced HRV Feature Optimization")
    print("This will improve upon your current best results using:")
    print("• Correlation analysis and collinearity removal")
    print("• Recursive feature elimination (RFE)")
    print("• Feature importance analysis") 
    print("• Advanced hyperparameter optimization")
    print("\nCurrent best: 68.75% with 12 features")
    print("Target: 70%+ accuracy")
    
    try:
        results = optimizer.run_advanced_optimization()
        
        if results:
            # Find the overall best result
            successful_results = [r for r in results if 'best_optimization' in r]
            if successful_results:
                best_overall = max(successful_results, key=lambda x: x['best_optimization']['accuracy'])
                best_opt = best_overall['best_optimization']
                
                print(f"\n{'='*80}")
                print("🏆 FINAL RECOMMENDATION")
                print("="*80)
                print(f"Best combination: {best_overall['name']}")
                print(f"Optimization method: {best_opt['method']}")
                print(f"Final accuracy: {best_opt['accuracy']:.4f}")
                print(f"Improvement over baseline: +{best_opt['improvement']*100:.2f}%")
                print(f"Number of features: {best_opt['n_features']}")
                print(f"Selected features: {best_opt['features']}")
                print(f"Optimal kernel: {best_opt['config']['kernel']}")
                print(f"Kernel parameters: {best_opt['config']['params']}")
                
                baseline_68_75 = 0.6875
                if best_opt['accuracy'] > baseline_68_75:
                    total_improvement = (best_opt['accuracy'] - baseline_68_75) * 100
                    print(f"\n🎉 TOTAL IMPROVEMENT: +{total_improvement:.2f}% over 68.75% baseline!")
                    
                    if best_opt['accuracy'] >= 0.70:
                        print("🎯 TARGET ACHIEVED: 70%+ accuracy reached!")
                else:
                    print(f"\n⚠️  No improvement over 68.75% baseline")
        
    except Exception as e:
        print(f"Error during advanced optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()