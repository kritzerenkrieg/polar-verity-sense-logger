def run_feature_optimization(self):
        """
        Run the complete feature optimization process.
        """
        print("=== HRV Feature Combination Optimization ===")
        print(f"Testing {len(self.predefined_combinations)} predefined combinations")
        
        # Step 1: Discover and load data
        print("\n1. Loading data...")
        file_metadata = self.discover_files()
        
        if not file_metadata:
            print("No valid CSV files found!")
            return
        
        self.all_data = self.load_data(file_metadata)
        
        # Debug step: Check available features
        self.debug_available_features()#!/usr/bin/env python3
"""
HRV-Based Lie Detection with Feature Combination Optimization
Tests multiple feature combinations to find the best performing set.
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
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class HRVFeatureOptimizer:
    def __init__(self, data_directory='.'):
        """
        Initialize the HRV Feature Optimization system.
        
        Args:
            data_directory (str): Directory containing CSV files
        """
        self.data_directory = data_directory
        
        # Define feature pools for systematic testing
        self.time_domain_features = [
            'nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
            'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index'
        ]
        self.frequency_domain_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
            'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak'
        ]
        
        # High-reliability features (baseline)
        self.baseline_features = {
            'time': ['nn_mean'],
            'frequency': ['lf_power', 'ln_hf', 'hf_norm', 'lf_norm', 'lf_hf_ratio']
        }
        
        # Predefined feature combinations to test (updated based on available features)
        self.predefined_combinations = [
            # Time domain only combinations (these should work)
            {'name': 'time_comprehensive', 'features': ['nn_mean', 'sdnn', 'rmssd', 'pnn50']},
            {'name': 'time_variability', 'features': ['sdnn', 'rmssd', 'pnn20', 'pnn50']},
            {'name': 'time_basic', 'features': ['nn_mean', 'sdnn', 'rmssd']},
            {'name': 'time_statistical', 'features': ['nn_mean', 'nn_min', 'nn_max', 'sdnn']},
            {'name': 'time_advanced', 'features': ['sdnn', 'sdsd', 'rmssd', 'triangular_index']},
            {'name': 'time_pnn_focus', 'features': ['rmssd', 'pnn20', 'pnn50']},
            {'name': 'time_complete', 'features': ['nn_mean', 'sdnn', 'rmssd', 'pnn20', 'pnn50', 'triangular_index']},
            
            # Single domain frequency (test if these work)
            {'name': 'freq_comprehensive', 'features': ['lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 'hf_norm']},
            {'name': 'freq_power_only', 'features': ['lf_power', 'hf_power', 'lf_hf_ratio']},
            {'name': 'freq_normalized', 'features': ['lf_norm', 'hf_norm', 'ln_hf']},
            {'name': 'freq_basic', 'features': ['lf_power', 'hf_power']},
            {'name': 'freq_ratios', 'features': ['lf_hf_ratio', 'lf_norm', 'hf_norm']},
            
            # Baseline (original combination)
            {'name': 'baseline', 'features': ['nn_mean', 'lf_power', 'ln_hf', 'hf_norm', 'lf_norm', 'lf_hf_ratio']},
            
            # Mixed combinations (may need debugging)
            {'name': 'mixed_essential', 'features': ['nn_mean', 'rmssd', 'lf_hf_ratio', 'hf_norm']},
            {'name': 'mixed_power', 'features': ['sdnn', 'lf_power', 'hf_power', 'ln_hf']},
            {'name': 'mixed_minimal', 'features': ['nn_mean', 'lf_power']},
            {'name': 'mixed_3_best', 'features': ['nn_mean', 'rmssd', 'lf_hf_ratio']},
            
            # Minimal combinations
            {'name': 'minimal_time_2', 'features': ['nn_mean', 'rmssd']},
            {'name': 'minimal_time_3', 'features': ['nn_mean', 'sdnn', 'rmssd']},
            {'name': 'minimal_variability', 'features': ['sdnn', 'rmssd']},
        ]
        
        # Simplified kernel grid (for faster training)
        self.kernel_configs = {
            'linear': {'svm__C': [0.1, 1, 10, 100]},
            'rbf': {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': ['scale', 0.01, 0.1]},
            'poly': {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 0.01], 'svm__degree': [2, 3]}
        }
        
        self.all_data = None
        self.results_summary = []
        
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
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        file_metadata = []
        
        for filename in csv_files:
            metadata = self.parse_filename(filename)
            if metadata:
                file_metadata.append(metadata)
                
        print(f"Discovered {len(file_metadata)} valid CSV files")
        
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
    
    def prepare_features_for_combination(self, df, feature_list):
        """
        Prepare feature matrix for a specific feature combination.
        
        Args:
            df (pd.DataFrame): Combined dataset
            feature_list (list): List of feature names to use
            
        Returns:
            tuple: (X, y, groups, available_features)
        """
        # Separate time and frequency data
        time_df = df[df['domain'] == 'time'].copy()
        freq_df = df[df['domain'] == 'frequency'].copy()
        
        # Debug: Check what columns are available
        print(f"  Time domain columns: {[col for col in time_df.columns if col in self.time_domain_features][:5]}...")
        print(f"  Freq domain columns: {[col for col in freq_df.columns if col in self.frequency_domain_features][:5]}...")
        
        # Determine merge columns
        merge_cols = ['subject', 'condition', 'label', 'binary_label']
        if 'window_number' in time_df.columns and 'window_number' in freq_df.columns:
            merge_cols.append('window_number')
        elif 'window_start' in time_df.columns and 'window_start' in freq_df.columns:
            merge_cols.extend(['window_start', 'window_end'])
        
        # Separate features by domain
        time_features_needed = [f for f in feature_list if f in self.time_domain_features]
        freq_features_needed = [f for f in feature_list if f in self.frequency_domain_features]
        
        print(f"  Time features needed: {time_features_needed}")
        print(f"  Freq features needed: {freq_features_needed}")
        
        # Check if we need both domains
        if len(time_features_needed) > 0 and len(freq_features_needed) > 0:
            # Need to merge both domains
            time_cols = merge_cols + [col for col in time_features_needed if col in time_df.columns]
            freq_cols = merge_cols + [col for col in freq_features_needed if col in freq_df.columns]
            
            # Ensure we have the required merge columns
            time_cols = [col for col in time_cols if col in time_df.columns]
            freq_cols = [col for col in freq_cols if col in freq_df.columns]
            
            # Check if we can merge
            common_merge_cols = [col for col in merge_cols if col in time_cols and col in freq_cols]
            if len(common_merge_cols) < 4:  # At least subject, condition, label, binary_label
                print(f"  ❌ Cannot merge: insufficient common columns")
                return None, None, None, []
            
            time_df_clean = time_df[time_cols].copy()
            freq_df_clean = freq_df[freq_cols].copy()
            
            # Merge datasets
            df_merged = pd.merge(
                time_df_clean, freq_df_clean, 
                on=common_merge_cols, 
                suffixes=('_time', '_freq')
            )
            
            # Collect available features from merged data
            available_features = []
            for feature in feature_list:
                if feature in time_features_needed and feature in df_merged.columns:
                    available_features.append(feature)
                elif feature in freq_features_needed and feature in df_merged.columns:
                    available_features.append(feature)
                elif f"{feature}_time" in df_merged.columns:
                    available_features.append(f"{feature}_time")
                elif f"{feature}_freq" in df_merged.columns:
                    available_features.append(f"{feature}_freq")
            
        elif len(time_features_needed) > 0:
            # Only time domain features
            available_features = [f for f in time_features_needed if f in time_df.columns]
            df_merged = time_df[merge_cols + available_features].copy()
            
        elif len(freq_features_needed) > 0:
            # Only frequency domain features
            available_features = [f for f in freq_features_needed if f in freq_df.columns]
            df_merged = freq_df[merge_cols + available_features].copy()
            
        else:
            print(f"  ❌ No recognized features in {feature_list}")
            return None, None, None, []
        
        # Final check for available features
        available_features = [f for f in available_features if f in df_merged.columns]
        
        if len(available_features) == 0:
            print(f"  ❌ No features found in merged data")
            return None, None, None, []
        
        # Prepare final dataset
        df_clean = df_merged.dropna(subset=available_features)
        
        if len(df_clean) == 0:
            print(f"  ❌ No data left after removing NaN values")
            return None, None, None, available_features
        
        X = df_clean[available_features].values
        y = df_clean['binary_label'].values
        groups = df_clean['subject'].values
        
        print(f"  ✅ Successfully prepared {X.shape[0]} samples with {len(available_features)} features")
        
        return X, y, groups, available_features
    
    def quick_svm_evaluation(self, X, y, groups, feature_combination_name):
        """
        Quick SVM evaluation using simplified grid search.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            feature_combination_name (str): Name of the feature combination
            
        Returns:
            dict: Results with best accuracy and configuration
        """
        logo = LeaveOneGroupOut()
        best_score = 0
        best_config = None
        
        # Test only the most promising kernels with reduced grid
        for kernel, param_grid in self.kernel_configs.items():
            try:
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
                    verbose=0  # Reduce verbosity for batch testing
                )
                
                grid_search.fit(X, y, groups=groups)
                
                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_config = {
                        'kernel': kernel,
                        'params': grid_search.best_params_,
                        'score': grid_search.best_score_
                    }
                    
            except Exception as e:
                print(f"Error with {kernel} kernel for {feature_combination_name}: {e}")
                continue
        
        return best_config
    
    def detailed_evaluation(self, X, y, groups, best_config, feature_names):
        """
        Perform detailed evaluation with the best configuration.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            best_config (dict): Best configuration from quick evaluation
            feature_names (list): List of feature names
            
        Returns:
            dict: Detailed evaluation results
        """
        logo = LeaveOneGroupOut()
        
        # Extract SVM parameters
        svm_params = {k.replace('svm__', ''): v for k, v in best_config['params'].items()}
        
        # Create pipeline with best parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**svm_params, kernel=best_config['kernel'], random_state=42))
        ])
        
        # Collect predictions for each fold
        y_true_all = []
        y_pred_all = []
        subject_results = {}
        
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = groups[test_idx][0]
            
            # Train and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Store results
            accuracy = accuracy_score(y_test, y_pred)
            subject_results[test_subject] = {
                'accuracy': accuracy,
                'n_samples': len(y_test),
                'true_labels': y_test.tolist(),
                'predictions': y_pred.tolist()
            }
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
        
        # Overall evaluation
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        conf_matrix = confusion_matrix(y_true_all, y_pred_all)
        
        results = {
            'overall_accuracy': overall_accuracy,
            'confusion_matrix': conf_matrix,
            'subject_results': subject_results,
            'best_config': best_config,
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }
        
        return results
    
    def test_feature_combination(self, combination_info):
        """
        Test a single feature combination.
        
        Args:
            combination_info (dict): Dictionary with 'name' and 'features'
            
        Returns:
            dict: Results for this combination
        """
        name = combination_info['name']
        features = combination_info['features']
        
        print(f"\n--- Testing {name} ({len(features)} features) ---")
        print(f"Features: {features}")
        
        # Prepare features
        X, y, groups, available_features = self.prepare_features_for_combination(self.all_data, features)
        
        if X is None or len(available_features) == 0:
            print(f"❌ No valid data for {name}")
            return {
                'name': name,
                'features': features,
                'available_features': [],
                'status': 'failed',
                'accuracy': 0.0,
                'n_samples': 0
            }
        
        print(f"Available features: {available_features}")
        print(f"Data shape: {X.shape}")
        
        # Quick evaluation
        best_config = self.quick_svm_evaluation(X, y, groups, name)
        
        if best_config is None:
            print(f"❌ No successful kernel for {name}")
            return {
                'name': name,
                'features': features,
                'available_features': available_features,
                'status': 'failed',
                'accuracy': 0.0,
                'n_samples': X.shape[0]
            }
        
        print(f"✅ Best: {best_config['kernel']} - {best_config['score']:.4f}")
        
        return {
            'name': name,
            'features': features,
            'available_features': available_features,
            'status': 'success',
            'accuracy': best_config['score'],
            'n_samples': X.shape[0],
            'best_kernel': best_config['kernel'],
            'best_params': best_config['params'],
            'n_features': len(available_features)
        }
    
    def debug_available_features(self):
        """Debug function to check what features are actually available in the data."""
        if self.all_data is None:
            return
        
        time_df = self.all_data[self.all_data['domain'] == 'time']
        freq_df = self.all_data[self.all_data['domain'] == 'frequency']
        
        print("\n=== DEBUGGING AVAILABLE FEATURES ===")
        print(f"Time domain data shape: {time_df.shape}")
        print(f"Frequency domain data shape: {freq_df.shape}")
        
        print(f"\nTime domain columns:")
        time_cols = [col for col in time_df.columns if col not in ['subject', 'condition', 'label', 'domain', 'binary_label', 'window_number', 'window_start', 'window_end']]
        for col in time_cols[:15]:  # Show first 15
            print(f"  {col}")
        if len(time_cols) > 15:
            print(f"  ... and {len(time_cols)-15} more")
        
        print(f"\nFrequency domain columns:")
        freq_cols = [col for col in freq_df.columns if col not in ['subject', 'condition', 'label', 'domain', 'binary_label', 'window_number', 'window_start', 'window_end']]
        for col in freq_cols[:15]:  # Show first 15
            print(f"  {col}")
        if len(freq_cols) > 15:
            print(f"  ... and {len(freq_cols)-15} more")
        
        print(f"\nExpected time domain features: {self.time_domain_features}")
        print(f"Expected frequency domain features: {self.frequency_domain_features}")
        
        # Check for matches
        time_matches = [col for col in self.time_domain_features if col in time_df.columns]
        freq_matches = [col for col in self.frequency_domain_features if col in freq_df.columns]
        
        print(f"\nMatched time domain features: {time_matches}")
        print(f"Matched frequency domain features: {freq_matches}")
        
        # Check sample data
        if len(time_df) > 0:
            print(f"\nSample time domain row:")
            sample_row = time_df.iloc[0]
            for col in time_matches[:5]:
                print(f"  {col}: {sample_row[col]}")
        
        if len(freq_df) > 0:
            print(f"\nSample frequency domain row:")
            sample_row = freq_df.iloc[0]
            for col in freq_matches[:5]:
                print(f"  {col}: {sample_row[col]}")
        
    def generate_combinatorial_features(self, max_features=6):
        """
        Generate all possible feature combinations up to max_features.
        
        Args:
            max_features (int): Maximum number of features in a combination
            
        Returns:
            list: List of feature combinations to test
        """
        all_features = self.time_domain_features + self.frequency_domain_features
        
        combinations_to_test = []
        
        # Generate combinations of different sizes
        for size in range(2, max_features + 1):
            print(f"Generating combinations of size {size}...")
            
            # Get all combinations of this size
            feature_combinations = list(combinations(all_features, size))
            
            # Limit to reasonable number for each size to keep runtime manageable
            if size <= 3:
                max_combinations = min(50, len(feature_combinations))  # All combinations for small sizes
            elif size == 4:
                max_combinations = min(100, len(feature_combinations))
            elif size == 5:
                max_combinations = min(50, len(feature_combinations))
            else:
                max_combinations = min(25, len(feature_combinations))
            
            # Sample combinations if too many
            if len(feature_combinations) > max_combinations:
                import random
                random.seed(42)  # For reproducibility
                selected_combinations = random.sample(feature_combinations, max_combinations)
            else:
                selected_combinations = feature_combinations
            
            # Convert to the format expected by test_feature_combination
            for i, combo in enumerate(selected_combinations):
                combinations_to_test.append({
                    'name': f'combo_{size}feat_{i+1:03d}',
                    'features': list(combo)
                })
            
            print(f"Added {len(selected_combinations)} combinations of size {size}")
        
        print(f"\nTotal combinations to test: {len(combinations_to_test)}")
        return combinations_to_test
    
    def run_combinatorial_optimization(self, max_features=6, quick_test=True):
        """
        Run combinatorial feature optimization.
        
        Args:
            max_features (int): Maximum number of features to test
            quick_test (bool): If True, use faster evaluation for screening
        """
        print("=== HRV Combinatorial Feature Optimization ===")
        print(f"Testing combinations up to {max_features} features")
        
        # Step 1: Discover and load data
        print("\n1. Loading data...")
        file_metadata = self.discover_files()
        
        if not file_metadata:
            print("No valid CSV files found!")
            return
        
        self.all_data = self.load_data(file_metadata)
        
        # Debug step: Check available features
        self.debug_available_features()
        
        # Step 2: Generate all combinations
        print(f"\n2. Generating feature combinations...")
        print("=" * 60)
        
        combinations_to_test = self.generate_combinatorial_features(max_features)
        
        # Step 3: Test all combinations
        print(f"\n3. Testing {len(combinations_to_test)} feature combinations...")
        print("=" * 80)
        
        results = []
        successful_count = 0
        
        for i, combination in enumerate(combinations_to_test):
            if i % 20 == 0 or i < 10:  # Show progress every 20 combinations, plus first 10
                print(f"\n[{i+1}/{len(combinations_to_test)}]", end="")
            else:
                print(".", end="", flush=True)  # Just show a dot for progress
            
            if i % 20 == 0 or i < 10:
                result = self.test_feature_combination(combination)
            else:
                # Simplified testing for bulk combinations
                result = self.test_feature_combination_quick(combination)
            
            results.append(result)
            
            if result['status'] == 'success':
                successful_count += 1
                
                # Show immediately if we find a very good result
                if result['accuracy'] > 0.70:
                    print(f"\n🎉 EXCELLENT RESULT: {result['name']} - {result['accuracy']:.4f}")
                    print(f"   Features: {result['available_features']}")
        
        print(f"\n\nCompleted testing. Successful combinations: {successful_count}/{len(combinations_to_test)}")
        
        # Step 4: Sort and display results
        print(f"\n4. COMBINATORIAL OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Sort by accuracy
        successful_results = [r for r in results if r['status'] == 'success']
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\nTOP 20 PERFORMING FEATURE COMBINATIONS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Name':<20} {'Accuracy':<8} {'#Feat':<5} {'Kernel':<8} {'Feature List'}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results[:20]):  # Top 20
            features_str = ', '.join(result['available_features'][:4])
            if len(result['available_features']) > 4:
                features_str += f" (+{len(result['available_features'])-4})"
            
            print(f"{i+1:<4} {result['name']:<20} {result['accuracy']:<8.4f} {result['n_features']:<5} {result.get('best_kernel', 'N/A'):<8} {features_str}")
        
        # Step 5: Detailed evaluation of top 5
        print(f"\n\n5. DETAILED EVALUATION OF TOP 5 COMBINATIONS")
        print("=" * 80)
        
        detailed_results = []
        
        for i, result in enumerate(successful_results[:5]):
            print(f"\n--- Detailed Analysis #{i+1}: {result['name']} ---")
            
            # Re-prepare features for detailed analysis
            X, y, groups, available_features = self.prepare_features_for_combination(
                self.all_data, result['features']
            )
            
            if X is not None:
                best_config = {
                    'kernel': result['best_kernel'],
                    'params': result['best_params'],
                    'score': result['accuracy']
                }
                
                detailed = self.detailed_evaluation(X, y, groups, best_config, available_features)
                detailed['combination_name'] = result['name']
                detailed['feature_list'] = result['features']
                
                print(f"Overall Accuracy: {detailed['overall_accuracy']:.4f}")
                print(f"Features ({len(available_features)}): {available_features}")
                print(f"Kernel: {result['best_kernel']}")
                print(f"Confusion Matrix:\n{detailed['confusion_matrix']}")
                
                detailed_results.append(detailed)
        
        # Step 6: Save results
        self.save_optimization_results(results, detailed_results)
        
        return results, detailed_results
    
    def test_feature_combination_quick(self, combination_info):
        """
        Quick test of a feature combination (minimal output for bulk testing).
        
        Args:
            combination_info (dict): Dictionary with 'name' and 'features'
            
        Returns:
            dict: Results for this combination
        """
        name = combination_info['name']
        features = combination_info['features']
        
        # Prepare features
        X, y, groups, available_features = self.prepare_features_for_combination(self.all_data, features)
        
        if X is None or len(available_features) == 0:
            return {
                'name': name,
                'features': features,
                'available_features': [],
                'status': 'failed',
                'accuracy': 0.0,
                'n_samples': 0
            }
        
        # Quick evaluation - only test RBF kernel for speed
        try:
            logo = LeaveOneGroupOut()
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
            ])
            
            # Simple cross-validation without grid search
            scores = []
            for train_idx, test_idx in logo.split(X, y, groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                scores.append(accuracy)
            
            avg_accuracy = np.mean(scores)
            
            return {
                'name': name,
                'features': features,
                'available_features': available_features,
                'status': 'success',
                'accuracy': avg_accuracy,
                'n_samples': X.shape[0],
                'best_kernel': 'rbf',
                'best_params': {'svm__C': 1.0, 'svm__gamma': 'scale'},
                'n_features': len(available_features)
            }
            
        except Exception as e:
            return {
                'name': name,
                'features': features,
                'available_features': available_features,
                'status': 'failed',
                'accuracy': 0.0,
                'n_samples': X.shape[0] if X is not None else 0
            }
    
    def run_feature_optimization(self):
        """
        Run the complete feature optimization process.
        """
        print("=== HRV Feature Combination Optimization ===")
        print(f"Testing {len(self.predefined_combinations)} predefined combinations")
        
        # Step 1: Discover and load data
        print("\n1. Loading data...")
        file_metadata = self.discover_files()
        
        if not file_metadata:
            print("No valid CSV files found!")
            return
        
        self.all_data = self.load_data(file_metadata)
        
        # Debug step: Check available features
        self.debug_available_features()
        
        # Step 2: Test all predefined combinations
        print(f"\n2. Testing feature combinations...")
        print("=" * 60)
        
        results = []
        
        for i, combination in enumerate(self.predefined_combinations):
            print(f"\n[{i+1}/{len(self.predefined_combinations)}]", end="")
            result = self.test_feature_combination(combination)
            results.append(result)
        
        # Step 3: Sort and display results
        print(f"\n\n3. OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Sort by accuracy
        successful_results = [r for r in results if r['status'] == 'success']
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\nTOP PERFORMING FEATURE COMBINATIONS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Name':<20} {'Accuracy':<8} {'Features':<8} {'Kernel':<8} {'Feature List'}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results[:10]):  # Top 10
            features_str = ', '.join(result['available_features'][:3])
            if len(result['available_features']) > 3:
                features_str += f" (+{len(result['available_features'])-3} more)"
            
            print(f"{i+1:<4} {result['name']:<20} {result['accuracy']:<8.4f} {result['n_features']:<8} {result.get('best_kernel', 'N/A'):<8} {features_str}")
        
        # Step 4: Detailed evaluation of top 3
        print(f"\n\n4. DETAILED EVALUATION OF TOP 3 COMBINATIONS")
        print("=" * 80)
        
        detailed_results = []
        
        for i, result in enumerate(successful_results[:3]):
            print(f"\n--- Detailed Analysis #{i+1}: {result['name']} ---")
            
            # Re-prepare features for detailed analysis
            X, y, groups, available_features = self.prepare_features_for_combination(
                self.all_data, result['features']
            )
            
            if X is not None:
                best_config = {
                    'kernel': result['best_kernel'],
                    'params': result['best_params'],
                    'score': result['accuracy']
                }
                
                detailed = self.detailed_evaluation(X, y, groups, best_config, available_features)
                detailed['combination_name'] = result['name']
                detailed['feature_list'] = result['features']
                
                print(f"Overall Accuracy: {detailed['overall_accuracy']:.4f}")
                print(f"Features ({len(available_features)}): {available_features}")
                print(f"Kernel: {result['best_kernel']}")
                print(f"Confusion Matrix:\n{detailed['confusion_matrix']}")
                
                detailed_results.append(detailed)
        
        # Step 5: Save results
        self.save_optimization_results(results, detailed_results)
        
        return results, detailed_results
        
        # Step 2: Test all predefined combinations
        print(f"\n2. Testing feature combinations...")
        print("=" * 60)
        
        results = []
        
        for i, combination in enumerate(self.predefined_combinations):
            print(f"\n[{i+1}/{len(self.predefined_combinations)}]", end="")
            result = self.test_feature_combination(combination)
            results.append(result)
        
        # Step 3: Sort and display results
        print(f"\n\n3. OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Sort by accuracy
        successful_results = [r for r in results if r['status'] == 'success']
        successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\nTOP PERFORMING FEATURE COMBINATIONS:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Name':<20} {'Accuracy':<8} {'Features':<8} {'Kernel':<8} {'Feature List'}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results[:10]):  # Top 10
            features_str = ', '.join(result['available_features'][:3])
            if len(result['available_features']) > 3:
                features_str += f" (+{len(result['available_features'])-3} more)"
            
            print(f"{i+1:<4} {result['name']:<20} {result['accuracy']:<8.4f} {result['n_features']:<8} {result.get('best_kernel', 'N/A'):<8} {features_str}")
        
        # Step 4: Detailed evaluation of top 3
        print(f"\n\n4. DETAILED EVALUATION OF TOP 3 COMBINATIONS")
        print("=" * 80)
        
        detailed_results = []
        
        for i, result in enumerate(successful_results[:3]):
            print(f"\n--- Detailed Analysis #{i+1}: {result['name']} ---")
            
            # Re-prepare features for detailed analysis
            X, y, groups, available_features = self.prepare_features_for_combination(
                self.all_data, result['features']
            )
            
            if X is not None:
                best_config = {
                    'kernel': result['best_kernel'],
                    'params': result['best_params'],
                    'score': result['accuracy']
                }
                
                detailed = self.detailed_evaluation(X, y, groups, best_config, available_features)
                detailed['combination_name'] = result['name']
                detailed['feature_list'] = result['features']
                
                print(f"Overall Accuracy: {detailed['overall_accuracy']:.4f}")
                print(f"Features ({len(available_features)}): {available_features}")
                print(f"Kernel: {result['best_kernel']}")
                print(f"Confusion Matrix:\n{detailed['confusion_matrix']}")
                
                detailed_results.append(detailed)
        
        # Step 5: Save results
        self.save_optimization_results(results, detailed_results)
        
        return results, detailed_results
    
    def save_optimization_results(self, quick_results, detailed_results):
        """Save optimization results to CSV files."""
        print(f"\n5. Saving results...")
        
        # Save quick results summary
        quick_df = pd.DataFrame([
            {
                'rank': i+1,
                'name': r['name'],
                'accuracy': r['accuracy'],
                'n_features': r['n_features'],
                'status': r['status'],
                'best_kernel': r.get('best_kernel', 'N/A'),
                'features': ', '.join(r['available_features']),
                'n_samples': r['n_samples']
            }
            for i, r in enumerate(sorted([r for r in quick_results if r['status'] == 'success'], 
                                       key=lambda x: x['accuracy'], reverse=True))
        ])
        
        quick_df.to_csv('feature_optimization_summary.csv', index=False)
        
        # Save detailed results for top combinations
        if detailed_results:
            detailed_summary = []
            for result in detailed_results:
                detailed_summary.append({
                    'combination_name': result['combination_name'],
                    'overall_accuracy': result['overall_accuracy'],
                    'n_features': result['n_features'],
                    'kernel': result['best_config']['kernel'],
                    'features': ', '.join(result['feature_names']),
                    'n_subjects': len(result['subject_results'])
                })
            
            detailed_df = pd.DataFrame(detailed_summary)
            detailed_df.to_csv('feature_optimization_detailed.csv', index=False)
        
        print("✅ Results saved to 'feature_optimization_summary.csv' and 'feature_optimization_detailed.csv'")

def main():
    """
    Main execution function for feature optimization.
    """
    optimizer = HRVFeatureOptimizer(data_directory='.')
    
    print("Starting HRV Feature Combination Optimization...")
    print("Choose optimization method:")
    print("1. Predefined combinations (fast, ~20 combinations)")
    print("2. Combinatorial approach (thorough, hundreds of combinations)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nRunning predefined combinations...")
            print("Current baseline: 67% accuracy with 6 features")
            print("\n" + "="*80)
            
            quick_results, detailed_results = optimizer.run_feature_optimization()
            
        elif choice == "2":
            print("\nRunning combinatorial optimization...")
            max_features = int(input("Maximum features per combination (recommended: 4-6): ") or "5")
            print(f"Testing all combinations up to {max_features} features")
            print("This may take 10-30 minutes depending on the number of combinations.")
            print("\n" + "="*80)
            
            quick_results, detailed_results = optimizer.run_combinatorial_optimization(
                max_features=max_features, quick_test=True
            )
            
        else:
            print("Invalid choice. Running predefined combinations...")
            quick_results, detailed_results = optimizer.run_feature_optimization()
        
        if detailed_results:
            best_result = detailed_results[0]
            print(f"\n{'='*80}")
            print("FINAL RECOMMENDATION")
            print("="*80)
            print(f"Best combination: {best_result['combination_name']}")
            print(f"Accuracy: {best_result['overall_accuracy']:.4f}")
            print(f"Number of features: {best_result['n_features']}")
            print(f"Features: {best_result['feature_names']}")
            print(f"Kernel: {best_result['best_config']['kernel']}")
            
            baseline_accuracy = 0.67
            if best_result['overall_accuracy'] > baseline_accuracy:
                improvement = (best_result['overall_accuracy'] - baseline_accuracy) * 100
                print(f"\n🎉 IMPROVEMENT: +{improvement:.1f}% over baseline!")
            else:
                print(f"\n⚠️  No improvement over baseline (67%)")
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()