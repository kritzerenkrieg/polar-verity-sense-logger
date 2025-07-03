#!/usr/bin/env python3
"""
HRV-Based Lie Detection - Feature Combination Testing
Tests specific high-reliability feature combinations to optimize accuracy
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class FeatureCombinationTester:
    def __init__(self, data_directory='.'):
        """
        Initialize the Feature Combination Tester.
        
        Args:
            data_directory (str): Directory containing CSV files
        """
        self.data_directory = data_directory
        
        # Define feature combinations to test
        self.feature_combinations = {
            'combo_1_90pct': ['lf_power', 'ln_hf'],           # 90% consistency features
            'combo_2_power': ['lf_power', 'lf_hf_ratio'],     # Power + ratio
            'combo_3_mixed': ['nn_mean', 'lf_norm'],          # Time + frequency mix  
            'combo_4_top3': ['lf_power', 'lf_norm', 'ln_hf'], # Top 3 features
        }
        
        # All available high-reliability features
        self.all_features = {
            'time': ['nn_mean'],
            'frequency': ['lf_power', 'ln_hf', 'hf_norm', 'lf_norm', 'lf_hf_ratio']
        }
        
        self.all_data = None
        self.results = {}
        
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
        """Discover and organize all CSV files."""
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
    
    def prepare_features_for_combination(self, df, feature_combination):
        """
        Prepare feature matrix for a specific combination.
        
        Args:
            df (pd.DataFrame): Combined dataset
            feature_combination (list): List of features to include
            
        Returns:
            tuple: (X, y, groups, available_features)
        """
        # Separate time and frequency features
        time_features = [f for f in feature_combination if f in self.all_features['time']]
        freq_features = [f for f in feature_combination if f in self.all_features['frequency']]
        
        if time_features and freq_features:
            # Need to merge time and frequency data
            time_df = df[df['domain'] == 'time'].copy()
            freq_df = df[df['domain'] == 'frequency'].copy()
            
            # Select only needed columns
            time_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + time_features
            freq_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + freq_features
            
            time_df_clean = time_df[time_cols].copy()
            freq_df_clean = freq_df[freq_cols].copy()
            
            # Merge on common identifiers
            merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
            df_filtered = pd.merge(
                time_df_clean, freq_df_clean, 
                on=merge_cols, 
                suffixes=('_time', '_freq')
            )
            
            all_feature_cols = time_features + freq_features
            
        elif freq_features:
            # Only frequency features
            df_filtered = df[df['domain'] == 'frequency'].copy()
            all_feature_cols = freq_features
            
        elif time_features:
            # Only time features
            df_filtered = df[df['domain'] == 'time'].copy()
            all_feature_cols = time_features
            
        else:
            raise ValueError("No valid features in combination")
        
        # Check for missing features
        available_features = [col for col in all_feature_cols if col in df_filtered.columns]
        missing_features = [col for col in all_feature_cols if col not in df_filtered.columns]
        
        if missing_features:
            print(f"Warning: Missing features {missing_features}")
        
        # Prepare final dataset
        df_clean = df_filtered.dropna(subset=available_features)
        
        X = df_clean[available_features].values
        y = df_clean['binary_label'].values
        groups = df_clean['subject'].values
        
        return X, y, groups, available_features
    
    def optimize_svm_for_combination(self, X, y, groups, combination_name):
        """
        Optimize SVM for a specific feature combination.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            combination_name (str): Name of the combination
            
        Returns:
            dict: Optimization results
        """
        print(f"\n=== OPTIMIZING {combination_name.upper()} ===")
        print(f"Feature matrix shape: {X.shape}")
        
        # Define parameter grids for different kernels
        param_grids = {
            'linear': {
                'svm__C': [0.01, 0.1, 1, 10, 100]
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
        
        # Test each kernel
        for kernel, param_grid in param_grids.items():
            print(f"Testing {kernel} kernel...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, class_weight='balanced', probability=True))
            ])
            
            # Grid search with LOSO CV
            try:
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=logo,
                    scoring='accuracy',
                    n_jobs=1
                )
                
                grid_search.fit(X, y, groups=groups)
                
                kernel_results[kernel] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                print(f"  {kernel}: {grid_search.best_score_:.4f} with {grid_search.best_params_}")
                
            except Exception as e:
                print(f"  {kernel}: Failed - {e}")
                continue
        
        return kernel_results
    
    def create_ensemble_for_combination(self, X, y, groups, kernel_results, combination_name):
        """
        Create ensemble model from best kernels for this combination.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            kernel_results (dict): Results from kernel optimization
            combination_name (str): Name of the combination
            
        Returns:
            tuple: (ensemble_model, ensemble_results)
        """
        if not kernel_results:
            print(f"No valid kernels for {combination_name}")
            return None, None
        
        print(f"\nCreating ensemble for {combination_name}...")
        
        # Sort kernels by performance
        sorted_kernels = sorted(kernel_results.items(), key=lambda x: x[1]['best_score'], reverse=True)
        
        # Create ensemble with top 2-3 performing kernels
        estimators = []
        for kernel, result in sorted_kernels[:3]:  # Top 3 kernels
            # Extract SVM parameters
            svm_params = {k.replace('svm__', ''): v for k, v in result['best_params'].items()}
            svm_params['kernel'] = kernel
            svm_params['class_weight'] = 'balanced'
            svm_params['probability'] = True
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(**svm_params))
            ])
            
            estimators.append((f'{kernel}_svm', pipeline))
        
        # Create voting classifier
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Evaluate ensemble with LOSO CV
        logo = LeaveOneGroupOut()
        y_true_all = []
        y_pred_all = []
        subject_results = {}
        
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = groups[test_idx][0]
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            
            # Store results
            accuracy = accuracy_score(y_test, y_pred)
            subject_results[test_subject] = {
                'accuracy': accuracy,
                'n_samples': len(y_test)
            }
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
        
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        
        ensemble_results = {
            'overall_accuracy': overall_accuracy,
            'subject_results': subject_results,
            'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
            'classification_report': classification_report(y_true_all, y_pred_all, target_names=['Truth', 'Lie']),
            'n_estimators': len(estimators),
            'kernels_used': [kernel for kernel, _ in sorted_kernels[:3]]
        }
        
        print(f"Ensemble accuracy: {overall_accuracy:.4f}")
        
        return ensemble, ensemble_results
    
    def run_combination_analysis(self):
        """Run complete feature combination analysis."""
        print("=== HRV LIE DETECTION - FEATURE COMBINATION TESTING ===")
        
        # Step 1: Load data
        print("\n1. Loading data...")
        file_metadata = self.discover_files()
        if not file_metadata:
            print("No valid CSV files found!")
            return
        
        self.all_data = self.load_data(file_metadata)
        
        # Step 2: Test each feature combination
        print("\n2. Testing feature combinations...")
        
        combination_results = {}
        
        for combo_name, features in self.feature_combinations.items():
            print(f"\n{'='*60}")
            print(f"TESTING {combo_name.upper()}: {features}")
            print(f"{'='*60}")
            
            try:
                # Prepare features
                X, y, groups, available_features = self.prepare_features_for_combination(
                    self.all_data, features
                )
                
                print(f"Available features: {available_features}")
                print(f"Samples: {len(X)}, Features: {X.shape[1]}")
                print(f"Label distribution: Truth={np.sum(y==0)}, Lie={np.sum(y==1)}")
                
                # Optimize SVM
                kernel_results = self.optimize_svm_for_combination(X, y, groups, combo_name)
                
                # Create ensemble
                ensemble, ensemble_results = self.create_ensemble_for_combination(
                    X, y, groups, kernel_results, combo_name
                )
                
                if ensemble_results:
                    combination_results[combo_name] = {
                        'features': available_features,
                        'kernel_results': kernel_results,
                        'ensemble_results': ensemble_results,
                        'ensemble_model': ensemble
                    }
                
            except Exception as e:
                print(f"Error testing {combo_name}: {e}")
                continue
        
        # Step 3: Compare results
        print(f"\n{'='*80}")
        print("FEATURE COMBINATION COMPARISON")
        print(f"{'='*80}")
        
        if combination_results:
            # Sort by accuracy
            sorted_results = sorted(
                combination_results.items(), 
                key=lambda x: x[1]['ensemble_results']['overall_accuracy'], 
                reverse=True
            )
            
            print(f"{'Rank':<4} {'Combination':<15} {'Features':<35} {'Accuracy':<10} {'Kernels'}")
            print("-" * 80)
            
            for rank, (combo_name, results) in enumerate(sorted_results, 1):
                features_str = ', '.join(results['features'])
                accuracy = results['ensemble_results']['overall_accuracy']
                kernels = ', '.join(results['ensemble_results']['kernels_used'])
                
                print(f"{rank:<4} {combo_name:<15} {features_str:<35} {accuracy:<10.4f} {kernels}")
            
            # Detailed results for best combination
            best_combo_name, best_results = sorted_results[0]
            print(f"\n{'='*60}")
            print(f"BEST COMBINATION: {best_combo_name.upper()}")
            print(f"{'='*60}")
            print(f"Features: {best_results['features']}")
            print(f"Overall Accuracy: {best_results['ensemble_results']['overall_accuracy']:.4f}")
            print(f"Kernels Used: {best_results['ensemble_results']['kernels_used']}")
            
            print(f"\nConfusion Matrix:")
            print(best_results['ensemble_results']['confusion_matrix'])
            
            print(f"\nSubject-wise Results:")
            subject_results = best_results['ensemble_results']['subject_results']
            for subject, result in sorted(subject_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                print(f"  {subject}: {result['accuracy']:.4f} ({result['n_samples']} samples)")
            
            # Save results
            self.save_combination_results(combination_results)
            
        else:
            print("No successful combinations tested!")
        
        return combination_results
    
    def save_combination_results(self, combination_results):
        """Save detailed results to CSV files."""
        print(f"\n3. Saving results...")
        
        # 1. Summary comparison
        summary_data = []
        for combo_name, results in combination_results.items():
            ensemble_res = results['ensemble_results']
            summary_data.append({
                'combination': combo_name,
                'features': ', '.join(results['features']),
                'n_features': len(results['features']),
                'overall_accuracy': ensemble_res['overall_accuracy'],
                'kernels_used': ', '.join(ensemble_res['kernels_used']),
                'n_estimators': ensemble_res['n_estimators']
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('overall_accuracy', ascending=False)
        summary_df.to_csv('feature_combination_summary.csv', index=False)
        
        # 2. Detailed subject results for each combination
        for combo_name, results in combination_results.items():
            subject_data = []
            for subject, result in results['ensemble_results']['subject_results'].items():
                subject_data.append({
                    'subject': subject,
                    'accuracy': result['accuracy'],
                    'n_samples': result['n_samples']
                })
            
            subject_df = pd.DataFrame(subject_data).sort_values('accuracy', ascending=False)
            subject_df.to_csv(f'subject_results_{combo_name}.csv', index=False)
        
        # 3. Kernel comparison for each combination
        for combo_name, results in combination_results.items():
            kernel_data = []
            for kernel, result in results['kernel_results'].items():
                kernel_data.append({
                    'kernel': kernel,
                    'best_score': result['best_score'],
                    'best_params': str(result['best_params'])
                })
            
            kernel_df = pd.DataFrame(kernel_data).sort_values('best_score', ascending=False)
            kernel_df.to_csv(f'kernel_results_{combo_name}.csv', index=False)
        
        print("Results saved to CSV files:")
        print("  - feature_combination_summary.csv (main comparison)")
        print("  - subject_results_[combination].csv (per-subject details)")
        print("  - kernel_results_[combination].csv (kernel optimization details)")

def main():
    """Main execution function."""
    tester = FeatureCombinationTester(data_directory='.')
    
    try:
        results = tester.run_combination_analysis()
        
        if results:
            best_combo = max(results.items(), key=lambda x: x[1]['ensemble_results']['overall_accuracy'])
            best_accuracy = best_combo[1]['ensemble_results']['overall_accuracy']
            
            print(f"\n🎯 ANALYSIS COMPLETE!")
            print(f"📊 Best Combination: {best_combo[0]}")
            print(f"📈 Best Accuracy: {best_accuracy:.1%}")
            
            if best_accuracy > 0.6:
                print(f"🚀 Excellent! Achieved >60% accuracy!")
            elif best_accuracy > 0.55:
                print(f"✅ Good improvement over baseline!")
            else:
                print(f"📋 Results saved for further analysis")
        else:
            print("❌ Analysis failed!")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()