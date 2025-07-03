#!/usr/bin/env python3
"""
Enhanced Global Model Selection with Detailed Subject-Level Logging
Removed subject limitation and added comprehensive CSV logging
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import warnings
from itertools import combinations
from collections import defaultdict
import time
import random
import threading
from contextlib import contextmanager
from datetime import datetime
warnings.filterwarnings('ignore')

@contextmanager
def timeout(seconds):
    """Cross-platform timeout context manager using threading."""
    class TimeoutException(Exception):
        pass
    
    def timeout_handler():
        raise TimeoutException("Operation timed out")
    
    # Use threading timer for cross-platform compatibility
    timer = threading.Timer(seconds, timeout_handler)
    timer.start()
    
    try:
        yield
    except TimeoutException:
        raise TimeoutError("Operation timed out")
    finally:
        timer.cancel()

class EnhancedCombinatorialModelSelector:
    def __init__(self, data_directory='.', min_features=2, max_features=6, 
                 max_combinations_per_size=50, early_stop_threshold=0.55,
                 max_eval_time=30):
        """
        Initialize the Enhanced Combinatorial Model Selection system.
        
        Args:
            data_directory: Directory containing CSV files
            min_features: Minimum number of features to test (default: 2)
            max_features: Maximum number of features to test (default: 6)
            max_combinations_per_size: Max combinations to test per feature count (default: 50)
            early_stop_threshold: Stop if accuracy below this threshold (default: 0.55)
            max_eval_time: Max time in seconds per combination evaluation (default: 30)
        """
        self.data_directory = data_directory
        self.min_features = min_features
        self.max_features = max_features
        self.max_combinations_per_size = max_combinations_per_size
        self.early_stop_threshold = early_stop_threshold
        self.max_eval_time = max_eval_time
        
        # All available features
        self.all_time_features = [
            'nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
            'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index'
        ]
        
        self.all_freq_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
            'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak'
        ]
        
        self.all_features = self.all_time_features + self.all_freq_features
        
        # Configuration space
        self.config_space = {
            'linear': [
                {'C': 0.1},
                {'C': 1.0},
                {'C': 10.0}
            ],
            'rbf': [
                {'C': 0.1, 'gamma': 'scale'}, 
                {'C': 1.0, 'gamma': 'scale'},
                {'C': 10.0, 'gamma': 'scale'},
                {'C': 1.0, 'gamma': 'auto'}
            ],
            'poly': [
                {'C': 1.0, 'degree': 2, 'gamma': 'scale'},
                {'C': 1.0, 'degree': 3, 'gamma': 'scale'}
            ]
        }
        
        self.all_data = None
        self.feature_importance_scores = None
        self.detailed_logs = []  # Store detailed evaluation logs
        
    def load_data(self):
        """Load and combine all CSV files."""
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        file_metadata = []
        
        for filename in csv_files:
            pattern = r'pr_([^_]+)_([^_]+)_(truth|lie)-sequence_(frequency_)?windowed\.csv'
            match = re.match(pattern, filename)
            
            if match:
                file_metadata.append({
                    'subject': match.group(1),
                    'condition': match.group(2),
                    'label': match.group(3),
                    'domain': 'frequency' if match.group(4) else 'time',
                    'filename': filename
                })
        
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
        
        if not all_dataframes:
            raise ValueError("No CSV files could be loaded")
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} total samples")
        print(f"Unique subjects: {sorted(combined_df['subject'].unique())}")
        
        return combined_df
    
    def compute_feature_importance(self, X, y):
        """Compute feature importance to guide combination selection."""
        print("Computing feature importance for smart sampling...")
        
        try:
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            # Add small random noise to break ties
            rf_importance = rf_importance + np.random.random(len(rf_importance)) * 0.001
            
            return rf_importance
            
        except Exception as e:
            print(f"Error computing feature importance: {e}")
            # Return uniform importance if calculation fails
            return np.ones(X.shape[1]) / X.shape[1]
    
    def generate_smart_feature_combinations(self, available_features, importance_scores):
        """Generate feature combinations using importance-guided sampling."""
        print(f"Generating smart feature combinations ({self.min_features} to {self.max_features} features)...")
        
        feature_importance_dict = dict(zip(available_features, importance_scores))
        
        all_combinations = []
        
        for n_features in range(self.min_features, self.max_features + 1):
            # Get all possible combinations for this feature count
            all_possible = list(combinations(available_features, n_features))
            
            if len(all_possible) <= self.max_combinations_per_size:
                # Use all combinations if we're under the limit
                selected_combinations = all_possible
            else:
                # Smart sampling: favor combinations with high-importance features
                
                # Score each combination by sum of feature importance
                combo_scores = []
                for combo in all_possible:
                    score = sum(feature_importance_dict[feat] for feat in combo)
                    combo_scores.append((combo, score))
                
                # Sort by score and take top combinations
                combo_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take top combinations
                selected_combinations = [combo for combo, score in combo_scores[:self.max_combinations_per_size]]
            
            # Convert to our format
            for i, combo in enumerate(selected_combinations):
                combo_name = f"combo_{n_features}feat_{i:03d}"
                all_combinations.append({
                    'name': combo_name,
                    'features': list(combo),
                    'n_features': n_features,
                    'importance_score': sum(feature_importance_dict[feat] for feat in combo)
                })
        
        # Sort by importance score within each feature count
        all_combinations.sort(key=lambda x: (x['n_features'], -x['importance_score']))
        
        print(f"Generated {len(all_combinations)} feature combinations")
        
        return all_combinations
    
    def prepare_features(self, feature_list):
        """Prepare feature matrix for a specific feature combination."""
        try:
            time_df = self.all_data[self.all_data['domain'] == 'time'].copy()
            freq_df = self.all_data[self.all_data['domain'] == 'frequency'].copy()
            
            merge_cols = ['subject', 'condition', 'label', 'binary_label']
            
            # Add window_number if it exists
            if 'window_number' in time_df.columns:
                merge_cols.append('window_number')
            
            time_needed = [f for f in feature_list if f in self.all_time_features]
            freq_needed = [f for f in feature_list if f in self.all_freq_features]
            
            if time_needed and freq_needed:
                time_cols = merge_cols + [f for f in time_needed if f in time_df.columns]
                freq_cols = merge_cols + [f for f in freq_needed if f in freq_df.columns]
                
                time_clean = time_df[time_cols].copy()
                freq_clean = freq_df[freq_cols].copy()
                
                merged_df = pd.merge(time_clean, freq_clean, on=merge_cols, how='inner')
                available_features = time_needed + freq_needed
                
            elif time_needed:
                available_features = [f for f in time_needed if f in time_df.columns]
                merged_df = time_df[merge_cols + available_features].copy()
                
            else:
                available_features = [f for f in freq_needed if f in freq_df.columns]
                merged_df = freq_df[merge_cols + available_features].copy()
            
            available_features = [f for f in available_features if f in merged_df.columns]
            
            if not available_features:
                return None, None, None, None, []
            
            # Drop rows with NaN values
            df_clean = merged_df.dropna(subset=available_features)
            
            if len(df_clean) < 10:  # Need minimum samples
                return None, None, None, None, []
            
            X = df_clean[available_features].values
            y = df_clean['binary_label'].values
            subjects = df_clean['subject'].values
            conditions = df_clean['condition'].values
            
            # Check for sufficient class balance
            if len(np.unique(y)) < 2:
                return None, None, None, None, []
            
            return X, y, subjects, conditions, available_features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None, None, None, None, []
    
    def create_optimized_configurations(self):
        """Create optimized SVM configurations to test."""
        all_configs = []
        
        for kernel, param_list in self.config_space.items():
            for params in param_list:
                config = {
                    'kernel': kernel,
                    'params': params,
                    'config_id': f"{kernel}_{hash(str(params))}"
                }
                all_configs.append(config)
        
        return all_configs
    
    def evaluate_subject_with_config(self, X, y, conditions, config):
        """Evaluate a single subject with a specific configuration."""
        unique_conditions = np.unique(conditions)
        
        if len(unique_conditions) < 2:
            return None
        
        try:
            # Use pipeline with scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use 3-fold CV for better accuracy estimation
            n_splits = min(3, len(unique_conditions))
            if n_splits < 2:
                return None
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, test_idx in skf.split(X_scaled, y):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                if len(np.unique(y_train)) < 2:
                    continue
                
                # Create SVM
                svm = SVC(kernel=config['kernel'], **config['params'], random_state=42)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
            
            return np.mean(accuracies) if accuracies else None
            
        except Exception as e:
            return None
    
    def comprehensive_global_evaluation(self, X, y, subjects, conditions, feature_names):
        """Comprehensive global evaluation with detailed logging."""
        try:
            # Use threading-based timeout for cross-platform compatibility
            result = [None]
            exception = [None]
            
            def evaluation_thread():
                try:
                    result[0] = self._comprehensive_evaluation_impl(X, y, subjects, conditions, feature_names)
                except Exception as e:
                    exception[0] = e
            
            # Start evaluation in separate thread
            thread = threading.Thread(target=evaluation_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.max_eval_time)
            
            if thread.is_alive():
                print(f"  TIMEOUT: Evaluation exceeded {self.max_eval_time} seconds")
                return None
            
            if exception[0]:
                print(f"  ERROR: {str(exception[0])[:50]}...")
                return None
                
            return result[0]
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:50]}...")
            return None
    
    def _comprehensive_evaluation_impl(self, X, y, subjects, conditions, feature_names):
        """Implementation of comprehensive evaluation with detailed logging."""
        unique_subjects = np.unique(subjects)
        all_configs = self.create_optimized_configurations()
        
        if len(unique_subjects) < 2:
            return None
        
        # REMOVED: Subject limitation - now use ALL subjects
        print(f"  Evaluating on {len(unique_subjects)} subjects: {list(unique_subjects)}")
        
        config_results = []
        
        for config in all_configs:
            subject_accuracies = []
            config_logs = []
            
            for subject in unique_subjects:
                subject_mask = subjects == subject
                subject_X = X[subject_mask]
                subject_y = y[subject_mask]
                subject_conditions = conditions[subject_mask]
                
                if len(subject_X) < 4:  # Need minimum samples
                    continue
                
                if len(np.unique(subject_conditions)) < 2:
                    continue
                
                if len(np.unique(subject_y)) < 2:
                    continue
                
                accuracy = self.evaluate_subject_with_config(
                    subject_X, subject_y, subject_conditions, config
                )
                
                if accuracy is not None:
                    subject_accuracies.append(accuracy)
                    
                    # Log detailed information
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'combination_name': None,  # Will be filled later
                        'features': ', '.join(feature_names),
                        'n_features': len(feature_names),
                        'kernel': config['kernel'],
                        'params': str(config['params']),
                        'subject': subject,
                        'accuracy': accuracy,
                        'n_samples': len(subject_X),
                        'n_conditions': len(np.unique(subject_conditions)),
                        'class_balance': f"{np.sum(subject_y == 0)}/{np.sum(subject_y == 1)}"
                    }
                    config_logs.append(log_entry)
            
            if subject_accuracies:
                mean_acc = np.mean(subject_accuracies)
                config_results.append({
                    'config': config,
                    'mean_accuracy': mean_acc,
                    'min_accuracy': np.min(subject_accuracies),
                    'max_accuracy': np.max(subject_accuracies),
                    'std_accuracy': np.std(subject_accuracies),
                    'n_subjects': len(subject_accuracies),
                    'subject_accuracies': subject_accuracies,
                    'detailed_logs': config_logs
                })
        
        # Return best config with all detailed logs
        if config_results:
            best_config = max(config_results, key=lambda x: x['mean_accuracy'])
            
            # Store all logs for this combination
            all_logs = []
            for config_result in config_results:
                all_logs.extend(config_result['detailed_logs'])
            
            best_config['all_detailed_logs'] = all_logs
            return best_config
        else:
            return None
    
    def run_comprehensive_optimization(self, save_top_n=5):
        """Run the comprehensive optimization process with detailed logging."""
        print("=== Enhanced Combinatorial Model Selection ===")
        print("With comprehensive subject evaluation and detailed logging")
        
        # Load data
        print("\nLoading data...")
        self.all_data = self.load_data()
        
        # Prepare full feature matrix for importance calculation
        print("Preparing features for importance analysis...")
        X_full, y_full, subjects_full, conditions_full, available_features = self.prepare_features(self.all_features)
        
        if X_full is None:
            print("Error: Could not prepare features")
            return []
        
        # Compute feature importance
        importance_scores = self.compute_feature_importance(X_full, y_full)
        
        # Generate smart feature combinations
        all_combinations = self.generate_smart_feature_combinations(available_features, importance_scores)
        
        print(f"\nTesting {len(all_combinations)} combinations")
        print(f"Timeout per combination: {self.max_eval_time} seconds")
        print(f"All subjects will be evaluated (no limitation)")
        
        all_results = []
        all_detailed_logs = []
        start_time = time.time()
        
        for i, combination in enumerate(all_combinations):
            print(f"\n[{i+1}/{len(all_combinations)}] Testing: {combination['name']}")
            print(f"  Features: {', '.join(combination['features'])}")
            
            # Prepare features for this combination
            X, y, subjects, conditions, feat_names = self.prepare_features(combination['features'])
            
            if X is None:
                print("  SKIP: Could not prepare features")
                continue
            
            # Check data quality
            if len(X) < 20:
                print("  SKIP: Insufficient samples")
                continue
            
            # Comprehensive evaluation with detailed logging
            combo_start = time.time()
            best_config = self.comprehensive_global_evaluation(X, y, subjects, conditions, feat_names)
            combo_time = time.time() - combo_start
            
            if best_config is None:
                print("  SKIP: No valid configuration found")
                continue
            
            # Update combination name in detailed logs
            for log_entry in best_config['all_detailed_logs']:
                log_entry['combination_name'] = combination['name']
            
            # Add to master detailed logs
            all_detailed_logs.extend(best_config['all_detailed_logs'])
            
            # Early stopping for clearly poor results
            if best_config['mean_accuracy'] < self.early_stop_threshold:
                print(f"  SKIP: Low accuracy ({best_config['mean_accuracy']:.3f} < {self.early_stop_threshold})")
                continue
            
            # Train final model
            try:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(
                        kernel=best_config['config']['kernel'],
                        **best_config['config']['params'],
                        random_state=42,
                        probability=True
                    ))
                ])
                
                pipeline.fit(X, y)
                
                # Create result
                result = {
                    'combination_name': combination['name'],
                    'n_features': len(feat_names),
                    'features': combination['features'],
                    'features_str': ', '.join(feat_names),
                    'global_mean_accuracy': best_config['mean_accuracy'],
                    'global_min_accuracy': best_config['min_accuracy'],
                    'global_max_accuracy': best_config['max_accuracy'],
                    'global_std_accuracy': best_config['std_accuracy'],
                    'n_subjects': best_config['n_subjects'],
                    'selected_config': best_config['config'],
                    'importance_score': combination['importance_score'],
                    'processing_time': combo_time,
                    'model_pipeline': pipeline,
                    'subject_accuracies': best_config['subject_accuracies']
                }
                
                all_results.append(result)
                
                print(f"  SUCCESS: {result['global_mean_accuracy']:.4f} ± {result['global_std_accuracy']:.4f} "
                      f"({result['selected_config']['kernel']}) [{combo_time:.1f}s] "
                      f"[{result['n_subjects']} subjects]")
                
            except Exception as e:
                print(f"  ERROR: Failed to train final model: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"\n=== OPTIMIZATION COMPLETE ===")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Successfully tested {len(all_results)} combinations")
        print(f"Generated {len(all_detailed_logs)} detailed log entries")
        
        if all_results:
            # Save results with detailed logs
            self.save_comprehensive_results(all_results, all_detailed_logs, save_top_n)
            
            # Show top results
            sorted_results = sorted(all_results, key=lambda x: x['global_mean_accuracy'], reverse=True)
            
            print(f"\n=== TOP {min(10, len(sorted_results))} RESULTS ===")
            for i, result in enumerate(sorted_results[:10]):
                print(f"{i+1:2d}. {result['global_mean_accuracy']:.4f} ± {result['global_std_accuracy']:.4f} "
                      f"({result['n_features']} feat) {result['selected_config']['kernel']} "
                      f"[{result['n_subjects']} subj] - {result['combination_name']}")
        
        return all_results
    
    def save_comprehensive_results(self, results, detailed_logs, save_top_n):
        """Save comprehensive optimization results with detailed logs."""
        print(f"\nSaving comprehensive results...")
        
        # Summary CSV
        summary_data = []
        for result in results:
            summary_data.append({
                'combination_name': result['combination_name'],
                'n_features': result['n_features'],
                'features': result['features_str'],
                'global_mean_accuracy': result['global_mean_accuracy'],
                'global_min_accuracy': result['global_min_accuracy'],
                'global_max_accuracy': result['global_max_accuracy'],
                'global_std_accuracy': result['global_std_accuracy'],
                'n_subjects': result['n_subjects'],
                'selected_kernel': result['selected_config']['kernel'],
                'selected_params': str(result['selected_config']['params']),
                'importance_score': result['importance_score'],
                'processing_time': result['processing_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('global_mean_accuracy', ascending=False)
        summary_df.to_csv('comprehensive_combinatorial_results.csv', index=False)
        print("SAVED: 'comprehensive_combinatorial_results.csv'")
        
        # Detailed logs CSV - This is the new feature!
        detailed_df = pd.DataFrame(detailed_logs)
        detailed_df = detailed_df.sort_values(['combination_name', 'kernel', 'subject'])
        detailed_df.to_csv('detailed_subject_accuracy_logs.csv', index=False)
        print("SAVED: 'detailed_subject_accuracy_logs.csv' - Contains all subject-level accuracies")
        
        # Subject-wise performance summary
        if detailed_logs:
            subject_summary = []
            for subject in sorted(set(log['subject'] for log in detailed_logs)):
                subject_logs = [log for log in detailed_logs if log['subject'] == subject]
                subject_summary.append({
                    'subject': subject,
                    'n_evaluations': len(subject_logs),
                    'mean_accuracy': np.mean([log['accuracy'] for log in subject_logs]),
                    'min_accuracy': np.min([log['accuracy'] for log in subject_logs]),
                    'max_accuracy': np.max([log['accuracy'] for log in subject_logs]),
                    'std_accuracy': np.std([log['accuracy'] for log in subject_logs]),
                    'best_combination': max(subject_logs, key=lambda x: x['accuracy'])['combination_name'],
                    'best_kernel': max(subject_logs, key=lambda x: x['accuracy'])['kernel'],
                    'best_accuracy': max(subject_logs, key=lambda x: x['accuracy'])['accuracy']
                })
            
            subject_summary_df = pd.DataFrame(subject_summary)
            subject_summary_df.to_csv('subject_performance_summary.csv', index=False)
            print("SAVED: 'subject_performance_summary.csv' - Per-subject performance summary")
        
        # Save top N models
        sorted_results = sorted(results, key=lambda x: x['global_mean_accuracy'], reverse=True)
        
        for i, result in enumerate(sorted_results[:save_top_n]):
            model_metadata = {
                'model_type': 'comprehensive_combinatorial_selection',
                'pipeline': result['model_pipeline'],
                'selected_features': result['features'],
                'model_config': {
                    'kernel': result['selected_config']['kernel'],
                    'params': result['selected_config']['params'],
                    'expected_mean_accuracy': result['global_mean_accuracy'],
                    'expected_min_accuracy': result['global_min_accuracy'],
                    'expected_max_accuracy': result['global_max_accuracy'],
                    'expected_std_accuracy': result['global_std_accuracy'],
                    'n_subjects_tested': result['n_subjects'],
                    'importance_score': result['importance_score'],
                    'method': 'comprehensive_combinatorial_selection'
                }
            }
            
            model_filename = f"comprehensive_top_{i+1:02d}_{result['combination_name']}.pkl"
            joblib.dump(model_metadata, model_filename)
            print(f"SAVED: '{model_filename}'")

def main():
    """Main execution function."""
    print("Enhanced Combinatorial Model Selection")
    print("With comprehensive subject evaluation and detailed logging")
    
    # Get user preferences
    try:
        min_feat = int(input("Minimum number of features to test (default: 2): ") or "2")
        max_feat = int(input("Maximum number of features to test (default: 4): ") or "4")
        max_per_size = int(input("Max combinations per feature count (default: 30): ") or "30")
        max_time = int(input("Max time per combination in seconds (default: 60): ") or "60")
    except ValueError:
        min_feat, max_feat, max_per_size, max_time = 2, 4, 30, 60
    
    selector = EnhancedCombinatorialModelSelector(
        data_directory='.', 
        min_features=min_feat, 
        max_features=max_feat,
        max_combinations_per_size=max_per_size,
        max_eval_time=max_time
    )
    
    try:
        results = selector.run_comprehensive_optimization()
        
        if results:
            best = max(results, key=lambda x: x['global_mean_accuracy'])
            print(f"\n=== BEST RESULT ===")
            print(f"Features: {best['features_str']}")
            print(f"Accuracy: {best['global_mean_accuracy']:.4f} ± {best['global_std_accuracy']:.4f}")
            print(f"Range: {best['global_min_accuracy']:.4f} - {best['global_max_accuracy']:.4f}")
            print(f"Config: {best['selected_config']['kernel']} {best['selected_config']['params']}")
            print(f"Evaluated on {best['n_subjects']} subjects")
            
            print(f"\n=== OUTPUT FILES ===")
            print("1. 'comprehensive_combinatorial_results.csv' - Summary of all combinations")
            print("2. 'detailed_subject_accuracy_logs.csv' - Complete subject-level accuracy logs")
            print("3. 'subject_performance_summary.csv' - Per-subject performance summary")
            print("4. Model files: 'comprehensive_top_XX_*.pkl'")
        else:
            print("No successful combinations found. Try reducing feature count or increasing timeout.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()