#!/usr/bin/env python3
"""
HRV-Based Lie Detection - Per-Subject Models with Leave-One-Condition-Out CV
Trains individual models for each subject using condition-based cross-validation
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
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PerSubjectHRVDetector:
    def __init__(self, data_directory='.'):
        """
        Initialize the Per-Subject HRV Lie Detection system.
        
        Args:
            data_directory (str): Directory containing CSV files
        """
        self.data_directory = data_directory
        
        # Best feature combinations from previous analysis
        self.feature_combinations = {
            'combo_1_90pct': ['lf_power', 'ln_hf'],           # 90% consistency features
            'combo_2_power': ['lf_power', 'lf_hf_ratio'],     # Power + ratio
            'combo_3_mixed': ['nn_mean', 'lf_norm'],          # Time + frequency mix  
            'combo_4_top3': ['lf_power', 'lf_norm', 'ln_hf'], # Top 3 features
            'combo_5_all': ['nn_mean', 'lf_power', 'ln_hf', 'hf_norm', 'lf_norm', 'lf_hf_ratio']  # All features
        }
        
        # Available features by domain
        self.all_features = {
            'time': ['nn_mean'],
            'frequency': ['lf_power', 'ln_hf', 'hf_norm', 'lf_norm', 'lf_hf_ratio']
        }
        
        self.all_data = None
        self.subject_models = {}
        self.subject_results = {}
        
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
        
        # Print data distribution
        subject_condition_counts = combined_df.groupby(['subject', 'condition', 'label']).size().unstack(fill_value=0)
        print(f"\nData distribution by subject and condition:")
        print(subject_condition_counts)
        
        return combined_df
    
    def prepare_subject_data(self, df, subject, feature_combination):
        """
        Prepare data for a specific subject and feature combination.
        
        Args:
            df (pd.DataFrame): Combined dataset
            subject (str): Subject ID
            feature_combination (list): List of features to include
            
        Returns:
            tuple: (X, y, conditions, available_features)
        """
        # Filter data for this subject only
        subject_df = df[df['subject'] == subject].copy()
        
        if len(subject_df) == 0:
            raise ValueError(f"No data found for subject {subject}")
        
        # Separate time and frequency features
        time_features = [f for f in feature_combination if f in self.all_features['time']]
        freq_features = [f for f in feature_combination if f in self.all_features['frequency']]
        
        if time_features and freq_features:
            # Need to merge time and frequency data
            time_df = subject_df[subject_df['domain'] == 'time'].copy()
            freq_df = subject_df[subject_df['domain'] == 'frequency'].copy()
            
            # Select only needed columns
            time_cols = ['condition', 'label', 'binary_label', 'window_number'] + time_features
            freq_cols = ['condition', 'label', 'binary_label', 'window_number'] + freq_features
            
            time_df_clean = time_df[time_cols].copy()
            freq_df_clean = freq_df[freq_cols].copy()
            
            # Merge on common identifiers
            merge_cols = ['condition', 'label', 'binary_label', 'window_number']
            df_filtered = pd.merge(
                time_df_clean, freq_df_clean, 
                on=merge_cols, 
                suffixes=('_time', '_freq')
            )
            
            all_feature_cols = time_features + freq_features
            
        elif freq_features:
            # Only frequency features
            df_filtered = subject_df[subject_df['domain'] == 'frequency'].copy()
            all_feature_cols = freq_features
            
        elif time_features:
            # Only time features
            df_filtered = subject_df[subject_df['domain'] == 'time'].copy()
            all_feature_cols = time_features
            
        else:
            raise ValueError("No valid features in combination")
        
        # Check for missing features
        available_features = [col for col in all_feature_cols if col in df_filtered.columns]
        missing_features = [col for col in all_feature_cols if col not in df_filtered.columns]
        
        if missing_features:
            print(f"Warning: Subject {subject} missing features {missing_features}")
        
        # Prepare final dataset
        df_clean = df_filtered.dropna(subset=available_features)
        
        X = df_clean[available_features].values
        y = df_clean['binary_label'].values
        conditions = df_clean['condition'].values
        
        return X, y, conditions, available_features
    
    def optimize_subject_model(self, X, y, conditions, subject, feature_combo_name):
        """
        Optimize SVM for a specific subject using Leave-One-Condition-Out CV.
        
        Args:
            X (np.array): Feature matrix for this subject
            y (np.array): Labels for this subject
            conditions (np.array): Condition groups for this subject
            subject (str): Subject ID
            feature_combo_name (str): Name of feature combination
            
        Returns:
            dict: Optimization results
        """
        print(f"\n=== OPTIMIZING MODEL FOR SUBJECT {subject.upper()} ===")
        print(f"Feature combination: {feature_combo_name}")
        print(f"Data shape: {X.shape}")
        print(f"Conditions: {np.unique(conditions)}")
        print(f"Label distribution: Truth={np.sum(y==0)}, Lie={np.sum(y==1)}")
        
        # Check if we have enough conditions for LOCO CV
        unique_conditions = np.unique(conditions)
        if len(unique_conditions) < 2:
            print(f"Subject {subject} has only {len(unique_conditions)} condition(s). Using simple train/test split.")
            return None
        
        # Parameter grids (simplified for faster execution per subject)
        param_grids = {
            'linear': {
                'svm__C': [0.1, 1, 10]
            },
            'rbf': {
                'svm__C': [0.1, 1, 10],
                'svm__gamma': ['scale', 'auto', 0.1]
            },
            'poly': {
                'svm__C': [0.1, 1, 10],
                'svm__gamma': ['scale'],
                'svm__degree': [2, 3]
            }
        }
        
        # Leave-One-Condition-Out CV
        loco = LeaveOneGroupOut()
        kernel_results = {}
        
        # Test each kernel
        for kernel, param_grid in param_grids.items():
            print(f"  Testing {kernel} kernel...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, class_weight='balanced', probability=True))
            ])
            
            try:
                # Grid search with LOCO CV
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=loco,
                    scoring='accuracy',
                    n_jobs=1
                )
                
                grid_search.fit(X, y, groups=conditions)
                
                kernel_results[kernel] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                print(f"    {kernel}: {grid_search.best_score_:.4f} with {grid_search.best_params_}")
                
            except Exception as e:
                print(f"    {kernel}: Failed - {e}")
                continue
        
        if not kernel_results:
            print(f"  All kernels failed for subject {subject}")
            return None
        
        # Find best kernel
        best_kernel = max(kernel_results.keys(), key=lambda k: kernel_results[k]['best_score'])
        best_result = kernel_results[best_kernel]
        
        print(f"  Best: {best_kernel} with accuracy {best_result['best_score']:.4f}")
        
        return {
            'subject': subject,
            'feature_combo': feature_combo_name,
            'kernel_results': kernel_results,
            'best_kernel': best_kernel,
            'best_params': best_result['best_params'],
            'best_score': best_result['best_score']
        }
    
    def evaluate_subject_model(self, X, y, conditions, optimization_result):
        """
        Evaluate the optimized model for a subject using LOCO CV.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            conditions (np.array): Condition groups
            optimization_result (dict): Results from optimization
            
        Returns:
            dict: Detailed evaluation results
        """
        if optimization_result is None:
            return None
        
        subject = optimization_result['subject']
        best_kernel = optimization_result['best_kernel']
        best_params = optimization_result['best_params']
        
        print(f"\nEvaluating {subject} with {best_kernel} kernel...")
        
        # Extract SVM parameters
        svm_params = {k.replace('svm__', ''): v for k, v in best_params.items()}
        svm_params['kernel'] = best_kernel
        svm_params['class_weight'] = 'balanced'
        
        # Create pipeline with best parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**svm_params))
        ])
        
        # LOCO CV evaluation
        loco = LeaveOneGroupOut()
        y_true_all = []
        y_pred_all = []
        condition_results = {}
        
        for train_idx, test_idx in loco.split(X, y, conditions):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_condition = conditions[test_idx][0]
            
            # Train and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Store results
            accuracy = accuracy_score(y_test, y_pred)
            condition_results[test_condition] = {
                'accuracy': accuracy,
                'n_samples': len(y_test),
                'true_labels': y_test.tolist(),
                'predictions': y_pred.tolist()
            }
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            print(f"  {test_condition}: {accuracy:.4f} ({len(y_test)} samples)")
        
        # Overall evaluation
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        
        return {
            'subject': subject,
            'overall_accuracy': overall_accuracy,
            'condition_results': condition_results,
            'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
            'classification_report': classification_report(y_true_all, y_pred_all, target_names=['Truth', 'Lie']),
            'optimization_info': optimization_result
        }
    
    def run_per_subject_analysis(self):
        """Run complete per-subject analysis."""
        print("=== HRV LIE DETECTION - PER-SUBJECT MODELS WITH LOCO CV ===")
        
        # Step 1: Load data
        print("\n1. Loading data...")
        file_metadata = self.discover_files()
        if not file_metadata:
            print("No valid CSV files found!")
            return
        
        self.all_data = self.load_data(file_metadata)
        
        # Get unique subjects
        subjects = sorted(self.all_data['subject'].unique())
        print(f"\nFound {len(subjects)} subjects: {subjects}")
        
        # Step 2: Test each feature combination across all subjects
        print("\n2. Testing feature combinations per subject...")
        
        all_results = defaultdict(dict)
        
        for combo_name, features in self.feature_combinations.items():
            print(f"\n{'='*80}")
            print(f"TESTING {combo_name.upper()}: {features}")
            print(f"{'='*80}")
            
            combo_results = {}
            
            for subject in subjects:
                try:
                    # Prepare subject data
                    X, y, conditions, available_features = self.prepare_subject_data(
                        self.all_data, subject, features
                    )
                    
                    print(f"\nSubject {subject}: {len(X)} samples, {X.shape[1]} features")
                    
                    # Check if subject has enough data
                    unique_conditions = np.unique(conditions)
                    if len(unique_conditions) < 2:
                        print(f"Skipping {subject}: insufficient conditions")
                        continue
                    
                    # Optimize model for this subject
                    optimization_result = self.optimize_subject_model(
                        X, y, conditions, subject, combo_name
                    )
                    
                    if optimization_result is None:
                        continue
                    
                    # Evaluate model
                    evaluation_result = self.evaluate_subject_model(
                        X, y, conditions, optimization_result
                    )
                    
                    if evaluation_result:
                        combo_results[subject] = evaluation_result
                        all_results[combo_name][subject] = evaluation_result
                        
                        print(f"  Final accuracy: {evaluation_result['overall_accuracy']:.4f}")
                
                except Exception as e:
                    print(f"Error with subject {subject} in {combo_name}: {e}")
                    continue
            
            # Summary for this combination
            if combo_results:
                accuracies = [r['overall_accuracy'] for r in combo_results.values()]
                print(f"\n{combo_name} Summary:")
                print(f"  Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
                print(f"  Best subject: {max(combo_results.items(), key=lambda x: x[1]['overall_accuracy'])}")
                print(f"  Subjects with >60%: {sum(1 for acc in accuracies if acc > 0.6)}/{len(accuracies)}")
        
        # Step 3: Overall comparison
        print(f"\n{'='*80}")
        print("OVERALL RESULTS COMPARISON")
        print(f"{'='*80}")
        
        self.compare_all_results(all_results)
        self.save_per_subject_results(all_results)
        
        return all_results
    
    def compare_all_results(self, all_results):
        """Compare results across all combinations and subjects."""
        print(f"\n{'Combination':<20} {'Mean Acc':<10} {'Std':<8} {'Best':<8} {'Subjects >60%':<15}")
        print("-" * 70)
        
        combination_summary = []
        
        for combo_name, combo_results in all_results.items():
            if not combo_results:
                continue
                
            accuracies = [r['overall_accuracy'] for r in combo_results.values()]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            best_acc = np.max(accuracies)
            high_performers = sum(1 for acc in accuracies if acc > 0.6)
            
            print(f"{combo_name:<20} {mean_acc:<10.4f} {std_acc:<8.4f} {best_acc:<8.4f} {high_performers}/{len(accuracies)}")
            
            combination_summary.append({
                'combination': combo_name,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'best_accuracy': best_acc,
                'subjects_above_60pct': high_performers,
                'total_subjects': len(accuracies)
            })
        
        # Find best overall combination
        if combination_summary:
            best_combo = max(combination_summary, key=lambda x: x['mean_accuracy'])
            print(f"\n🏆 BEST COMBINATION: {best_combo['combination']}")
            print(f"📊 Mean Accuracy: {best_combo['mean_accuracy']:.1%}")
            print(f"🎯 High Performers: {best_combo['subjects_above_60pct']}/{best_combo['total_subjects']} subjects >60%")
    
    def save_per_subject_results(self, all_results):
        """Save detailed results to CSV files."""
        print(f"\n3. Saving results...")
        
        # 1. Overall summary
        summary_data = []
        for combo_name, combo_results in all_results.items():
            for subject, result in combo_results.items():
                summary_data.append({
                    'combination': combo_name,
                    'subject': subject,
                    'overall_accuracy': result['overall_accuracy'],
                    'best_kernel': result['optimization_info']['best_kernel'],
                    'best_params': str(result['optimization_info']['best_params'])
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('per_subject_summary.csv', index=False)
        
        # 2. Detailed condition results
        condition_data = []
        for combo_name, combo_results in all_results.items():
            for subject, result in combo_results.items():
                for condition, cond_result in result['condition_results'].items():
                    condition_data.append({
                        'combination': combo_name,
                        'subject': subject,
                        'condition': condition,
                        'accuracy': cond_result['accuracy'],
                        'n_samples': cond_result['n_samples']
                    })
        
        if condition_data:
            condition_df = pd.DataFrame(condition_data)
            condition_df.to_csv('per_subject_condition_details.csv', index=False)
        
        # 3. Best results per subject
        best_results = {}
        for combo_name, combo_results in all_results.items():
            for subject, result in combo_results.items():
                if subject not in best_results or result['overall_accuracy'] > best_results[subject]['accuracy']:
                    best_results[subject] = {
                        'subject': subject,
                        'best_combination': combo_name,
                        'accuracy': result['overall_accuracy'],
                        'kernel': result['optimization_info']['best_kernel']
                    }
        
        if best_results:
            best_df = pd.DataFrame(list(best_results.values()))
            best_df = best_df.sort_values('accuracy', ascending=False)
            best_df.to_csv('best_per_subject_models.csv', index=False)
        
        print("Results saved to CSV files:")
        print("  - per_subject_summary.csv (all combinations × subjects)")
        print("  - per_subject_condition_details.csv (condition-level results)")
        print("  - best_per_subject_models.csv (best model per subject)")

def main():
    """Main execution function."""
    detector = PerSubjectHRVDetector(data_directory='.')
    
    try:
        results = detector.run_per_subject_analysis()
        
        if results:
            # Calculate overall statistics
            all_accuracies = []
            for combo_results in results.values():
                for subject_result in combo_results.values():
                    all_accuracies.append(subject_result['overall_accuracy'])
            
            if all_accuracies:
                mean_acc = np.mean(all_accuracies)
                high_performers = sum(1 for acc in all_accuracies if acc > 0.6)
                very_high = sum(1 for acc in all_accuracies if acc > 0.7)
                
                print(f"\n🎯 PER-SUBJECT ANALYSIS COMPLETE!")
                print(f"📊 Average Accuracy: {mean_acc:.1%}")
                print(f"🚀 Models >60%: {high_performers}/{len(all_accuracies)}")
                print(f"⭐ Models >70%: {very_high}/{len(all_accuracies)}")
                
                if mean_acc > 0.6:
                    print(f"🎉 SUCCESS! Per-subject approach significantly improved accuracy!")
                elif high_performers > len(all_accuracies) * 0.3:
                    print(f"✅ PROMISING! Many subjects show good individual performance!")
                else:
                    print(f"📋 Results show individual variation patterns")
        else:
            print("❌ Analysis failed!")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()