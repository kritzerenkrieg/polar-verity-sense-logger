#!/usr/bin/env python3
"""
Improved HRV Feature Optimization with Better Hyperparameter Search
Addresses the issues in the original code for more effective optimization.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

class ImprovedHRVOptimizer:
    def __init__(self, data_directory='.'):
        """Initialize the Improved HRV Optimization system."""
        self.data_directory = data_directory
        
        # Top performing combinations from your results
        self.top_combinations = [
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
                'name': 'forward_8feat',
                'features': ['lf_power', 'pnn20', 'nn_min', 'sdsd', 'nn_count', 'sdnn', 'hf_power', 'lf_norm'],
                'baseline_accuracy': 0.7023
            }
        ]
        
        # EXPANDED hyperparameter search space
        self.kernel_configs = {
            'linear': {
                'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'svm__class_weight': [None, 'balanced']
            },
            'rbf': {
                'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
                'svm__class_weight': [None, 'balanced']
            },
            'poly': {
                'svm__C': [0.01, 0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'svm__degree': [2, 3, 4],
                'svm__class_weight': [None, 'balanced']
            },
            'sigmoid': {
                'svm__C': [0.01, 0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'svm__class_weight': [None, 'balanced']
            }
        }
        
        self.all_data = None
        self.all_results = []
        
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
        
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} total samples")
        
        # Print class distribution
        class_dist = combined_df['binary_label'].value_counts()
        print(f"Class distribution: Truth={class_dist[0]}, Lie={class_dist[1]}")
        
        return combined_df
    
    def prepare_features(self, feature_list):
        """Prepare feature matrix for a specific feature combination."""
        time_df = self.all_data[self.all_data['domain'] == 'time'].copy()
        freq_df = self.all_data[self.all_data['domain'] == 'frequency'].copy()
        
        time_features = ['nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
                        'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index']
        freq_features = ['lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
                        'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak']
        
        merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
        
        time_needed = [f for f in feature_list if f in time_features]
        freq_needed = [f for f in feature_list if f in freq_features]
        
        if time_needed and freq_needed:
            time_cols = merge_cols + [f for f in time_needed if f in time_df.columns]
            freq_cols = merge_cols + [f for f in freq_needed if f in freq_df.columns]
            
            time_clean = time_df[time_cols].copy()
            freq_clean = freq_df[freq_cols].copy()
            
            merged_df = pd.merge(time_clean, freq_clean, on=merge_cols)
            available_features = time_needed + freq_needed
            
        elif time_needed:
            available_features = [f for f in time_needed if f in time_df.columns]
            merged_df = time_df[merge_cols + available_features].copy()
            
        else:
            available_features = [f for f in freq_needed if f in freq_df.columns]
            merged_df = freq_df[merge_cols + available_features].copy()
        
        available_features = [f for f in available_features if f in merged_df.columns]
        df_clean = merged_df.dropna(subset=available_features)
        
        if len(df_clean) == 0 or len(available_features) == 0:
            return None, None, None, []
        
        X = df_clean[available_features].values
        y = df_clean['binary_label'].values
        groups = df_clean['subject'].values
        
        return X, y, groups, available_features
    
    def optimize_hyperparameters_improved(self, X, y, groups):
        """
        Improved hyperparameter optimization with multiple strategies.
        """
        print(f"   Optimizing with {len(np.unique(groups))} subjects, {X.shape[0]} samples")
        
        # Strategy 1: Use Stratified K-Fold for hyperparameter search (more stable)
        # Strategy 2: Use LOGO for final evaluation (subject-independent)
        
        best_configs = []
        
        # Use StratifiedKFold for hyperparameter search (more data per fold)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for kernel, param_grid in self.kernel_configs.items():
            try:
                print(f"     Testing {kernel} kernel...")
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(kernel=kernel, random_state=42))
                ])
                
                # Hyperparameter search with StratifiedKFold
                grid_search = GridSearchCV(
                    pipeline, 
                    param_grid, 
                    cv=skf,  # Use StratifiedKFold instead of LOGO
                    scoring='accuracy', 
                    n_jobs=-1, 
                    verbose=0
                )
                
                grid_search.fit(X, y)  # No groups parameter for StratifiedKFold
                
                # Evaluate best config with LOGO for subject-independent validation
                logo = LeaveOneGroupOut()
                logo_scores = []
                
                best_pipeline = grid_search.best_estimator_
                
                for train_idx, test_idx in logo.split(X, y, groups):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    best_pipeline.fit(X_train, y_train)
                    score = best_pipeline.score(X_test, y_test)
                    logo_scores.append(score)
                
                logo_mean_score = np.mean(logo_scores)
                logo_std_score = np.std(logo_scores)
                
                config = {
                    'kernel': kernel,
                    'params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,  # StratifiedKFold score
                    'logo_score': logo_mean_score,        # LOGO score
                    'logo_std': logo_std_score,
                    'score': logo_mean_score  # Use LOGO score for comparison
                }
                
                best_configs.append(config)
                
                print(f"       CV Score: {grid_search.best_score_:.4f}")
                print(f"       LOGO Score: {logo_mean_score:.4f} ± {logo_std_score:.4f}")
                print(f"       Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"     Error with {kernel} kernel: {e}")
                continue
        
        if not best_configs:
            print("     Warning: All kernels failed, using default config")
            return {
                'kernel': 'rbf',
                'params': {'svm__C': 1.0, 'svm__gamma': 'scale'},
                'cv_score': 0.5,
                'logo_score': 0.5,
                'logo_std': 0.0,
                'score': 0.5
            }
        
        # Return best config based on LOGO score
        best_config = max(best_configs, key=lambda x: x['score'])
        print(f"     Best kernel: {best_config['kernel']} (LOGO: {best_config['score']:.4f})")
        
        return best_config
    
    def detailed_evaluation(self, X, y, groups, config, feature_names, combination_name):
        """Detailed evaluation with per-subject results."""
        logo = LeaveOneGroupOut()
        
        # Extract SVM parameters
        svm_params = {k.replace('svm__', ''): v for k, v in config['params'].items()}
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**svm_params, kernel=config['kernel'], random_state=42))
        ])
        
        detailed_results = []
        subject_summaries = []
        
        y_true_all = []
        y_pred_all = []
        
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = groups[test_idx][0]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Per-sample results
            for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
                detailed_results.append({
                    'combination_name': combination_name,
                    'kernel': config['kernel'],
                    'kernel_params': str(config['params']),
                    'features': ', '.join(feature_names),
                    'n_features': len(feature_names),
                    'subject': test_subject,
                    'sample_idx': i,
                    'true_label': 'truth' if true_label == 0 else 'lie',
                    'predicted_label': 'truth' if pred_label == 0 else 'lie',
                    'correct': true_label == pred_label,
                    'subject_accuracy': accuracy
                })
            
            # Subject summary
            subject_summaries.append({
                'combination_name': combination_name,
                'kernel': config['kernel'],
                'kernel_params': str(config['params']),
                'features': ', '.join(feature_names),
                'n_features': len(feature_names),
                'subject': test_subject,
                'accuracy': accuracy,
                'n_samples': len(y_test),
                'n_correct': sum(y_test == y_pred),
                'n_incorrect': sum(y_test != y_pred),
                'cv_score': config.get('cv_score', 0),
                'logo_std': config.get('logo_std', 0)
            })
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
        
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        
        return {
            'overall_accuracy': overall_accuracy,
            'detailed_results': detailed_results,
            'subject_summaries': subject_summaries,
            'config': config,
            'feature_names': feature_names
        }
    
    def feature_importance_analysis(self, X, y, feature_names):
        """Enhanced feature importance using Random Forest."""
        try:
            rf = RandomForestClassifier(
                n_estimators=200,  # More trees for stability
                max_depth=5,       # Prevent overfitting
                min_samples_split=5,
                random_state=42
            )
            rf.fit(StandardScaler().fit_transform(X), y)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        except Exception as e:
            print(f"Feature importance analysis failed: {e}")
            return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
    
    def top_k_features(self, X, y, feature_names, k=6):
        """Select top k features using univariate selection."""
        try:
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_names)))
            X_selected = selector.fit_transform(X, y)
            
            selected_mask = selector.get_support()
            selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
            
            # Get feature scores for analysis
            scores = selector.scores_
            feature_scores = pd.DataFrame({
                'feature': feature_names,
                'score': scores,
                'selected': selected_mask
            }).sort_values('score', ascending=False)
            
            print(f"     Top univariate features: {selected_features}")
            
            return selected_features
        except Exception as e:
            print(f"Feature selection failed: {e}")
            return feature_names[:k]
    
    def save_trained_model(self, best_result, X, y):
        """Train and save the best model as a PKL file."""
        print(f"\n=== Training and Saving Best Model ===")
        
        try:
            import ast
            kernel_params_dict = ast.literal_eval(best_result['kernel_params'])
            svm_params = {k.replace('svm__', ''): v for k, v in kernel_params_dict.items()}
            
            optimized_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    kernel=best_result['kernel'],
                    **svm_params,
                    random_state=42,
                    probability=True
                ))
            ])
            
            print(f"Training model with:")
            print(f"  - Kernel: {best_result['kernel']}")
            print(f"  - Parameters: {svm_params}")
            print(f"  - Features: {best_result['n_features']}")
            print(f"  - Training samples: {len(X)}")
            
            optimized_pipeline.fit(X, y)
            
            features_list = best_result['features'].split(', ')
            model_metadata = {
                'pipeline': optimized_pipeline,
                'selected_features': features_list,
                'model_config': {
                    'kernel': best_result['kernel'],
                    'kernel_params': kernel_params_dict,
                    'svm_params': svm_params,
                    'expected_accuracy': best_result['optimized_accuracy'],
                    'method': best_result['method'],
                    'combination_name': best_result['combination_name'],
                    'n_features': best_result['n_features'],
                    'training_samples': len(X),
                    'improvement_percent': best_result['improvement_percent'],
                    'cv_score': best_result.get('cv_score', 0),
                    'logo_std': best_result.get('logo_std', 0)
                },
                'feature_domains': {
                    'time_features': [f for f in features_list 
                                    if f in ['nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
                                           'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index']],
                    'freq_features': [f for f in features_list
                                    if f in ['lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
                                           'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak']]
                }
            }
            
            model_filename = f'best_hrv_model_improved_{best_result["combination_name"]}.pkl'
            joblib.dump(model_metadata, model_filename)
            
            print(f"SAVED: '{model_filename}'")
            print(f"  - Expected accuracy: {best_result['optimized_accuracy']:.4f}")
            print(f"  - Improvement: {best_result['improvement_percent']:+.2f}%")
            print(f"  - CV Score: {best_result.get('cv_score', 0):.4f}")
            print(f"  - LOGO Std: {best_result.get('logo_std', 0):.4f}")
            
            return model_filename
            
        except Exception as e:
            print(f"Error saving trained model: {e}")
            return None
    
    def run_optimization(self):
        """Run the improved optimization process."""
        print("=== Improved HRV Advanced Optimization ===")
        
        # Load data
        print("Loading data...")
        self.all_data = self.load_data()
        
        all_detailed_results = []
        all_subject_summaries = []
        combination_results = []
        
        for i, combination in enumerate(self.top_combinations):
            print(f"\n[{i+1}/{len(self.top_combinations)}] Optimizing: {combination['name']}")
            print(f"Features: {combination['features']}")
            print(f"Baseline: {combination['baseline_accuracy']:.4f}")
            
            # Prepare features
            X, y, groups, available_features = self.prepare_features(combination['features'])
            
            if X is None:
                print(f"ERROR: Could not prepare features for {combination['name']}")
                continue
            
            print(f"Data prepared: {X.shape[0]} samples, {len(available_features)} features")
            
            # Test different optimization approaches
            approaches = []
            
            # 1. Original features with improved hyperparameter optimization
            try:
                config = self.optimize_hyperparameters_improved(X, y, groups)
                approaches.append({
                    'method': 'improved_hyperparameter_optimization',
                    'features': available_features,
                    'config': config,
                    'accuracy': config['score']
                })
                print(f"   Improved hyperparameter optimization: {config['score']:.4f}")
            except Exception as e:
                print(f"   Improved hyperparameter optimization failed: {e}")
            
            # 2. Feature importance + hyperparameter optimization
            try:
                importance_df = self.feature_importance_analysis(X, y, available_features)
                top_important = importance_df.head(min(6, len(available_features)))['feature'].tolist()
                
                if len(top_important) >= 3:  # Ensure minimum features
                    X_imp, y_imp, groups_imp, _ = self.prepare_features(top_important)
                    if X_imp is not None:
                        config_imp = self.optimize_hyperparameters_improved(X_imp, y_imp, groups_imp)
                        approaches.append({
                            'method': 'importance_based_selection',
                            'features': top_important,
                            'config': config_imp,
                            'accuracy': config_imp['score']
                        })
                        print(f"   Importance-based selection: {config_imp['score']:.4f}")
            except Exception as e:
                print(f"   Importance-based selection failed: {e}")
            
            # 3. Univariate feature selection + hyperparameter optimization
            try:
                top_features = self.top_k_features(X, y, available_features, k=6)
                if len(top_features) >= 3:
                    X_top, y_top, groups_top, _ = self.prepare_features(top_features)
                    if X_top is not None:
                        config_top = self.optimize_hyperparameters_improved(X_top, y_top, groups_top)
                        approaches.append({
                            'method': 'univariate_feature_selection',
                            'features': top_features,
                            'config': config_top,
                            'accuracy': config_top['score']
                        })
                        print(f"   Univariate feature selection: {config_top['score']:.4f}")
            except Exception as e:
                print(f"   Univariate feature selection failed: {e}")
            
            # Find best approach
            if approaches:
                best_approach = max(approaches, key=lambda x: x['accuracy'])
                
                print(f"   Best approach: {best_approach['method']} - {best_approach['accuracy']:.4f}")
                
                # Get detailed evaluation
                X_best, y_best, groups_best, features_best = self.prepare_features(best_approach['features'])
                if X_best is not None:
                    detailed = self.detailed_evaluation(
                        X_best, y_best, groups_best, best_approach['config'], 
                        features_best, combination['name']
                    )
                    
                    all_detailed_results.extend(detailed['detailed_results'])
                    all_subject_summaries.extend(detailed['subject_summaries'])
                    
                    # Record combination result
                    combination_results.append({
                        'combination_name': combination['name'],
                        'method': best_approach['method'],
                        'kernel': best_approach['config']['kernel'],
                        'kernel_params': str(best_approach['config']['params']),
                        'features': ', '.join(features_best),
                        'n_features': len(features_best),
                        'baseline_accuracy': combination['baseline_accuracy'],
                        'optimized_accuracy': detailed['overall_accuracy'],
                        'improvement': detailed['overall_accuracy'] - combination['baseline_accuracy'],
                        'improvement_percent': (detailed['overall_accuracy'] - combination['baseline_accuracy']) * 100,
                        'cv_score': best_approach['config'].get('cv_score', 0),
                        'logo_std': best_approach['config'].get('logo_std', 0)
                    })
                    
                    # Store data for model training
                    combination_results[-1]['training_data'] = {
                        'X': X_best,
                        'y': y_best,
                        'groups': groups_best,
                        'features': features_best
                    }
        
        # Save results
        self.save_results(combination_results, all_detailed_results, all_subject_summaries)
        
        return combination_results
    
    def save_results(self, combination_results, detailed_results, subject_summaries):
        """Save all results to CSV files."""
        print("\n=== Saving Results ===")
        
        # 1. Combination results
        if combination_results:
            combo_df = pd.DataFrame(combination_results)
            combo_df = combo_df.sort_values('optimized_accuracy', ascending=False)
            combo_df.to_csv('feature_kernel_combinations_improved.csv', index=False)
            print("SAVED: 'feature_kernel_combinations_improved.csv'")
            
            # Print summary
            print(f"\nTop Results:")
            for _, row in combo_df.head(3).iterrows():
                print(f"  {row['combination_name']}: {row['optimized_accuracy']:.4f} "
                     f"({row['improvement_percent']:+.2f}%) - {row['method']}")
                print(f"    CV: {row.get('cv_score', 0):.4f}, LOGO Std: {row.get('logo_std', 0):.4f}")
        
        # 2. Detailed results
        if detailed_results:
            detail_df = pd.DataFrame(detailed_results)
            detail_df.to_csv('per_subject_detailed_results_improved.csv', index=False)
            print("SAVED: 'per_subject_detailed_results_improved.csv'")
        
        # 3. Subject summaries
        if subject_summaries:
            summary_df = pd.DataFrame(subject_summaries)
            summary_df.to_csv('per_subject_summary_results_improved.csv', index=False)
            print("SAVED: 'per_subject_summary_results_improved.csv'")
        
        # 4. Train and save best model
        if combination_results:
            best_result = max(combination_results, key=lambda x: x['optimized_accuracy'])
            
            if 'training_data' in best_result:
                training_data = best_result['training_data']
                model_filename = self.save_trained_model(
                    best_result, 
                    training_data['X'], 
                    training_data['y']
                )
                
                if model_filename:
                    print(f"\nREADY FOR PRODUCTION:")
                    print(f"  Model file: {model_filename}")
                    print(f"  Expected accuracy: {best_result['optimized_accuracy']:.4f}")
                    print(f"  Features: {best_result['features']}")

def main():
    """Main execution function."""
    optimizer = ImprovedHRVOptimizer(data_directory='.')
    
    print("Improved HRV Advanced Optimization")
    print("Enhanced hyperparameter search, multiple CV strategies, expanded parameter space")
    
    try:
        results = optimizer.run_optimization()
        
        if results:
            best = max(results, key=lambda x: x['optimized_accuracy'])
            print(f"\nBEST RESULT:")
            print(f"   Combination: {best['combination_name']}")
            print(f"   Method: {best['method']}")
            print(f"   Accuracy: {best['optimized_accuracy']:.4f}")
            print(f"   Improvement: {best['improvement_percent']:+.2f}%")
            print(f"   Features: {best['n_features']}")
            print(f"   CV Score: {best.get('cv_score', 0):.4f}")
            print(f"   LOGO Std: {best.get('logo_std', 0):.4f}")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()