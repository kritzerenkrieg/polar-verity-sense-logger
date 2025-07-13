#!/usr/bin/env python3
"""
Focused Feature Combination Trainer
Trains SVM models using only the recommended feature combinations from analysis
Creates comprehensive summary of each combination's performance on each subject
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class FocusedCombinationTrainer:
    def __init__(self, data_directory='.'):
        """
        Focused trainer using only recommended feature combinations
        
        Args:
            data_directory: Directory containing CSV files
        """
        self.data_directory = data_directory
        
        # Core feature sets for time and frequency domains
        self.core_time_features = [
            'nn_mean', 'sdnn', 'rmssd', 'pnn50', 'triangular_index', 'nn_min'
        ]
        
        self.core_freq_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 'hf_norm', 'lf_peak'
        ]
        
        self.all_features = self.core_time_features + self.core_freq_features
        
        # RECOMMENDED FEATURE SET (the 6 most frequently appearing features)
        self.recommended_features = ['hf_norm', 'lf_hf_ratio', 'sdnn', 'lf_norm', 'rmssd', 'lf_power']
        
        # Single combination using all 6 recommended features
        self.recommended_combinations = [
            self.recommended_features  # Use all 6 features together
        ]
        
        # RECOMMENDED MODEL PARAMETERS (from your analysis)
        self.recommended_configs = [
            {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 2.0, 'gamma': 'scale'},    # Best avg: 0.9429
            {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},    # Second: 0.9416
            {'kernel': 'rbf', 'C': 0.5, 'gamma': 'scale'},    # Third best
        ]
        
        self.all_data = None
        self.detailed_results = []
        
    def load_data(self):
        """Load and combine all CSV files."""
        print("📂 Loading CSV files...")
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
        
        print(f"✓ Loaded {len(combined_df)} samples from {len(np.unique(combined_df['subject']))} subjects")
        
        # Show class balance per subject
        print("\n📊 Class balance per subject:")
        for subject in sorted(combined_df['subject'].unique()):
            subj_data = combined_df[combined_df['subject'] == subject]
            truth_count = len(subj_data[subj_data['binary_label'] == 0])
            lie_count = len(subj_data[subj_data['binary_label'] == 1])
            total = len(subj_data)
            balance = min(truth_count, lie_count) / max(truth_count, lie_count) if max(truth_count, lie_count) > 0 else 0
            print(f"  {subject}: {truth_count}T/{lie_count}L ({total} total, balance: {balance:.2f})")
        
        return combined_df
    
    def prepare_features(self, feature_list, data_subset=None):
        """Prepare feature matrix with robust error handling."""
        if data_subset is None:
            data_subset = self.all_data
            
        try:
            time_df = data_subset[data_subset['domain'] == 'time'].copy()
            freq_df = data_subset[data_subset['domain'] == 'frequency'].copy()
            
            merge_cols = ['subject', 'condition', 'label', 'binary_label']
            
            # Add window_number if it exists
            if 'window_number' in time_df.columns:
                merge_cols.append('window_number')
            
            time_needed = [f for f in feature_list if f in self.core_time_features]
            freq_needed = [f for f in feature_list if f in self.core_freq_features]
            
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
            
            if len(df_clean) < 6:
                return None, None, None, None, []
            
            X = df_clean[available_features].values
            y = df_clean['binary_label'].values
            subjects = df_clean['subject'].values
            conditions = df_clean['condition'].values
            
            # Check for sufficient class balance
            if len(np.unique(y)) < 2:
                return None, None, None, None, []
            
            class_counts = np.bincount(y)
            if np.min(class_counts) < 2:
                return None, None, None, None, []
            
            return X, y, subjects, conditions, available_features
            
        except Exception as e:
            return None, None, None, None, []
    
    def evaluate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics."""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            
            try:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            except:
                precision = accuracy
                recall = accuracy
                f1 = accuracy
            
            # Balanced score combining accuracy and F1
            balanced_score = (accuracy * 0.6) + (f1 * 0.4)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'balanced_score': balanced_score
            }
        except:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'balanced_score': 0.0
            }
    
    def train_and_evaluate_combination(self, subject_id, feature_combo, config):
        """Train and evaluate a specific feature combination on a subject."""
        try:
            # Get subject's data
            subject_data = self.all_data[self.all_data['subject'] == subject_id]
            
            # Check minimum data requirements
            if len(subject_data) < 8:
                return None
            
            # Check class balance
            class_counts = subject_data['binary_label'].value_counts()
            if len(class_counts) < 2 or class_counts.min() < 2:
                return None
            
            # Prepare features
            X, y, _, _, available_features = self.prepare_features(
                feature_combo, data_subset=subject_data
            )
            
            if X is None or len(X) < 6:
                return None
            
            # Cross-validation
            n_splits = min(3, len(np.unique(y)))
            if n_splits < 2:
                return None
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            cv_scores = []
            cv_detailed_metrics = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                if len(np.unique(y_train_fold)) < 2:
                    continue
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)
                
                # Train SVM
                svm = SVC(**config, random_state=42)
                svm.fit(X_train_scaled, y_train_fold)
                
                y_pred = svm.predict(X_val_scaled)
                
                # Evaluate
                metrics = self.evaluate_metrics(y_val_fold, y_pred)
                cv_scores.append(metrics['balanced_score'])
                cv_detailed_metrics.append(metrics)
            
            if not cv_scores:
                return None
            
            # Calculate mean metrics across folds
            mean_metrics = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'balanced_score']:
                mean_metrics[f'{metric}_mean'] = np.mean([m[metric] for m in cv_detailed_metrics])
                mean_metrics[f'{metric}_std'] = np.std([m[metric] for m in cv_detailed_metrics])
            
            # Train final model on full data for model saving
            scaler_full = StandardScaler()
            X_scaled_full = scaler_full.fit_transform(X)
            svm_full = SVC(**config, random_state=42, probability=True)
            svm_full.fit(X_scaled_full, y)
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(**config, random_state=42, probability=True))
            ])
            pipeline.fit(X, y)
            
            result = {
                'subject_id': subject_id,
                'feature_combination': '|'.join(sorted(feature_combo)),
                'features_list': ', '.join(sorted(feature_combo)),
                'n_features': len(feature_combo),
                'config_string': f"{config['kernel']}_C{config['C']}_gamma{config['gamma']}",
                'kernel': config['kernel'],
                'C': config['C'],
                'gamma': config['gamma'],
                'n_samples': len(X),
                'class_distribution': f"{np.sum(y == 0)}/{np.sum(y == 1)}",
                'n_cv_folds': len(cv_scores),
                'trained_model': pipeline,
                'available_features': available_features,
                **mean_metrics
            }
            
            return result
            
        except Exception as e:
            return None
    
    def evaluate_cross_subject(self, model_result, test_subjects):
        """Evaluate model across other subjects."""
        cross_results = []
        
        for test_subject in test_subjects:
            if test_subject == model_result['subject_id']:
                continue
            
            try:
                # Get test subject's data
                test_data = self.all_data[self.all_data['subject'] == test_subject]
                
                # Extract original feature combination from the result
                original_features = model_result['features_list'].split(', ')
                
                X_test, y_test, _, _, _ = self.prepare_features(
                    original_features, data_subset=test_data
                )
                
                if X_test is None or len(X_test) < 4:
                    continue
                
                # Test the model
                y_pred = model_result['trained_model'].predict(X_test)
                metrics = self.evaluate_metrics(y_test, y_pred)
                
                cross_result = {
                    'model_subject': model_result['subject_id'],
                    'test_subject': test_subject,
                    'feature_combination': model_result['feature_combination'],
                    'config_string': model_result['config_string'],
                    'n_test_samples': len(X_test),
                    'test_class_distribution': f"{np.sum(y_test == 0)}/{np.sum(y_test == 1)}",
                    **metrics
                }
                
                cross_results.append(cross_result)
                
            except Exception as e:
                continue
        
        return cross_results
    
    def run_focused_training(self):
        """Run training with focused feature combinations."""
        print("=== Focused Feature Combination Training ===")
        print(f"🎯 Using the 6 recommended features: {', '.join(self.recommended_features)}")
        print(f"⚙️  Using {len(self.recommended_configs)} recommended SVM configurations")
        
        # Load data
        self.all_data = self.load_data()
        all_subjects = sorted(self.all_data['subject'].unique())
        
        print(f"\n🔬 Training on {len(all_subjects)} subjects")
        print(f"📊 Total experiments: {len(all_subjects)} × 1 combination × {len(self.recommended_configs)} configs = {len(all_subjects) * len(self.recommended_configs)}")
        
        # STEP 1: Train all combinations on all subjects
        print(f"\n🧪 STEP 1: Training all combinations...")
        
        all_results = []
        experiment_count = 0
        total_experiments = len(all_subjects) * len(self.recommended_configs)
        
        for subject_id in all_subjects:
            print(f"\n📋 Subject: {subject_id}")
            subject_results = []
            
            # Use the single recommended feature combination
            feature_combo = self.recommended_features
            
            for config_idx, config in enumerate(self.recommended_configs):
                experiment_count += 1
                
                print(f"  [{experiment_count:3d}/{total_experiments}] Features: {len(feature_combo)}, Config: {config['kernel']}-C{config['C']}", end=" ")
                
                result = self.train_and_evaluate_combination(subject_id, feature_combo, config)
                
                if result:
                    all_results.append(result)
                    subject_results.append(result)
                    print(f"✓ Score: {result['balanced_score_mean']:.4f}")
                else:
                    print("❌ Failed")
            
            if subject_results:
                best_subject_result = max(subject_results, key=lambda x: x['balanced_score_mean'])
                print(f"  🏆 Best for {subject_id}: {best_subject_result['balanced_score_mean']:.4f} (Config: {best_subject_result['config_string']})")
            else:
                print(f"  ❌ No successful models for {subject_id}")
        
        print(f"\n✓ Completed {len(all_results)} successful experiments out of {total_experiments}")
        
        if not all_results:
            print("❌ No successful experiments!")
            return None
        
        # STEP 2: Cross-subject evaluation
        print(f"\n🎯 STEP 2: Cross-subject evaluation...")
        
        all_cross_results = []
        
        for result in all_results:
            cross_results = self.evaluate_cross_subject(result, all_subjects)
            all_cross_results.extend(cross_results)
        
        print(f"✓ Completed {len(all_cross_results)} cross-subject evaluations")
        
        # STEP 3: Analysis and summaries
        print(f"\n📊 STEP 3: Creating comprehensive summaries...")
        
        self.create_comprehensive_summaries(all_results, all_cross_results, all_subjects)
        
        return {
            'all_results': all_results,
            'cross_results': all_cross_results,
            'subjects': all_subjects
        }
    
    def create_comprehensive_summaries(self, all_results, all_cross_results, all_subjects):
        """Create comprehensive summaries of all experiments."""
        
        # 1. DETAILED RESULTS BY SUBJECT AND COMBINATION
        detailed_df = pd.DataFrame(all_results)
        detailed_df = detailed_df.sort_values(['subject_id', 'balanced_score_mean'], ascending=[True, False])
        
        # Remove the trained_model and available_features columns for CSV export
        export_detailed = detailed_df.drop(['trained_model', 'available_features'], axis=1, errors='ignore')
        export_detailed.to_csv('focused_detailed_results.csv', index=False)
        print("✓ 'focused_detailed_results.csv' - Detailed results for every subject-combination-config")
        
        # 2. SUMMARY BY FEATURE COMBINATION (across all subjects) - now just one combination
        combo_str = '|'.join(sorted(self.recommended_features))
        combo_results = [r for r in all_results if r['feature_combination'] == combo_str]
        
        if combo_results:
            scores = [r['balanced_score_mean'] for r in combo_results]
            accuracies = [r['accuracy_mean'] for r in combo_results]
            f1_scores = [r['f1_mean'] for r in combo_results]
            
            combo_summary_data = [{
                'feature_combination': combo_str,
                'features_list': ', '.join(sorted(self.recommended_features)),
                'n_features': len(self.recommended_features),
                'n_successful_subjects': len(combo_results),
                'balanced_score_mean': np.mean(scores),
                'balanced_score_std': np.std(scores),
                'balanced_score_min': np.min(scores),
                'balanced_score_max': np.max(scores),
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'success_rate': len(combo_results) / len(all_subjects)
            }]
            
            combo_summary_df = pd.DataFrame(combo_summary_data)
            combo_summary_df.to_csv('focused_combination_summary.csv', index=False)
            print("✓ 'focused_combination_summary.csv' - Performance summary for the recommended feature set")
        
        # 3. SUMMARY BY SUBJECT (across all combinations)
        subject_summary_data = []
        
        for subject in all_subjects:
            subject_results = [r for r in all_results if r['subject_id'] == subject]
            
            if subject_results:
                best_result = max(subject_results, key=lambda x: x['balanced_score_mean'])
                scores = [r['balanced_score_mean'] for r in subject_results]
                
                subject_summary_data.append({
                    'subject_id': subject,
                    'n_successful_combinations': len(subject_results),
                    'best_balanced_score': best_result['balanced_score_mean'],
                    'best_accuracy': best_result['accuracy_mean'],
                    'best_f1': best_result['f1_mean'],
                    'best_features': best_result['features_list'],
                    'best_config': best_result['config_string'],
                    'avg_balanced_score': np.mean(scores),
                    'std_balanced_score': np.std(scores),
                    'success_rate': len(subject_results) / len(self.recommended_configs)
                })
            else:
                subject_summary_data.append({
                    'subject_id': subject,
                    'n_successful_combinations': 0,
                    'best_balanced_score': 0.0,
                    'best_accuracy': 0.0,
                    'best_f1': 0.0,
                    'best_features': 'None',
                    'best_config': 'None',
                    'avg_balanced_score': 0.0,
                    'std_balanced_score': 0.0,
                    'success_rate': 0.0
                })
        
        subject_summary_df = pd.DataFrame(subject_summary_data)
        subject_summary_df = subject_summary_df.sort_values('best_balanced_score', ascending=False)
        subject_summary_df.to_csv('focused_subject_summary.csv', index=False)
        print("✓ 'focused_subject_summary.csv' - Performance summary by subject")
        
        # 4. CROSS-SUBJECT EVALUATION RESULTS
        if all_cross_results:
            cross_df = pd.DataFrame(all_cross_results)
            cross_df.to_csv('focused_cross_subject_evaluation.csv', index=False)
            print("✓ 'focused_cross_subject_evaluation.csv' - Cross-subject evaluation results")
            
            # 5. CROSS-SUBJECT GENERALIZATION SUMMARY
            cross_summary_data = []
            
            for result in all_results:
                model_id = f"{result['subject_id']}_{result['feature_combination']}_{result['config_string']}"
                relevant_cross = [c for c in all_cross_results 
                                if c['model_subject'] == result['subject_id'] 
                                and c['feature_combination'] == result['feature_combination']
                                and c['config_string'] == result['config_string']]
                
                if relevant_cross:
                    cross_scores = [c['balanced_score'] for c in relevant_cross]
                    cross_accuracies = [c['accuracy'] for c in relevant_cross]
                    
                    cross_summary_data.append({
                        'model_id': model_id,
                        'model_subject': result['subject_id'],
                        'feature_combination': result['feature_combination'],
                        'config_string': result['config_string'],
                        'self_performance': result['balanced_score_mean'],
                        'cross_subject_mean': np.mean(cross_scores),
                        'cross_subject_std': np.std(cross_scores),
                        'cross_subject_min': np.min(cross_scores),
                        'cross_subject_max': np.max(cross_scores),
                        'cross_accuracy_mean': np.mean(cross_accuracies),
                        'n_cross_tests': len(relevant_cross),
                        'generalization_gap': result['balanced_score_mean'] - np.mean(cross_scores)
                    })
            
            cross_summary_df = pd.DataFrame(cross_summary_data)
            cross_summary_df = cross_summary_df.sort_values('cross_subject_mean', ascending=False)
            cross_summary_df.to_csv('focused_generalization_summary.csv', index=False)
            print("✓ 'focused_generalization_summary.csv' - Generalization performance summary")
        
        # 6. CONFIGURATION PERFORMANCE SUMMARY
        config_summary_data = []
        
        for config in self.recommended_configs:
            config_str = f"{config['kernel']}_C{config['C']}_gamma{config['gamma']}"
            config_results = [r for r in all_results if r['config_string'] == config_str]
            
            if config_results:
                scores = [r['balanced_score_mean'] for r in config_results]
                
                config_summary_data.append({
                    'config_string': config_str,
                    'kernel': config['kernel'],
                    'C': config['C'],
                    'gamma': config['gamma'],
                    'n_successful_models': len(config_results),
                    'balanced_score_mean': np.mean(scores),
                    'balanced_score_std': np.std(scores),
                    'balanced_score_min': np.min(scores),
                    'balanced_score_max': np.max(scores),
                    'success_rate': len(config_results) / (len(all_subjects))
                })
        
        config_summary_df = pd.DataFrame(config_summary_data)
        config_summary_df = config_summary_df.sort_values('balanced_score_mean', ascending=False)
        config_summary_df.to_csv('focused_config_summary.csv', index=False)
        print("✓ 'focused_config_summary.csv' - Performance summary by SVM configuration")
        
        # 7. BEST OVERALL MODEL
        best_overall = max(all_results, key=lambda x: x['balanced_score_mean'])
        
        # Save the best model
        joblib.dump({
            'method': 'focused_combination_training',
            'best_model_info': {
                'subject_id': best_overall['subject_id'],
                'features': best_overall['features_list'].split(', '),
                'config': {
                    'kernel': best_overall['kernel'],
                    'C': best_overall['C'],
                    'gamma': best_overall['gamma']
                },
                'performance': {
                    'balanced_score': best_overall['balanced_score_mean'],
                    'accuracy': best_overall['accuracy_mean'],
                    'f1': best_overall['f1_mean']
                }
            },
            'trained_model': best_overall['trained_model'],
            'timestamp': datetime.now().isoformat()
        }, 'focused_best_model.pkl')
        print("✓ 'focused_best_model.pkl' - Best performing model")
        
        # 8. OVERALL SUMMARY
        overall_summary = {
            'method': 'focused_combination_training',
            'total_experiments': len(all_results),
            'total_possible_experiments': len(all_subjects) * len(self.recommended_configs),
            'success_rate': len(all_results) / (len(all_subjects) * len(self.recommended_configs)),
            'n_subjects': len(all_subjects),
            'n_feature_combinations': 1,
            'n_configs': len(self.recommended_configs),
            'best_overall_score': best_overall['balanced_score_mean'],
            'best_overall_subject': best_overall['subject_id'],
            'best_overall_features': best_overall['features_list'],
            'best_overall_config': best_overall['config_string'],
            'avg_performance': np.mean([r['balanced_score_mean'] for r in all_results]),
            'std_performance': np.std([r['balanced_score_mean'] for r in all_results]),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_df = pd.DataFrame([overall_summary])
        summary_df.to_csv('focused_overall_summary.csv', index=False)
        print("✓ 'focused_overall_summary.csv' - Overall training summary")
        
        # Print key findings
        print(f"\n🏆 KEY FINDINGS:")
        print(f"   Best overall performance: {best_overall['balanced_score_mean']:.4f}")
        print(f"   Best subject: {best_overall['subject_id']}")
        print(f"   Best features: {best_overall['features_list']}")
        print(f"   Best config: {best_overall['config_string']}")
        print(f"   Success rate: {len(all_results) / (len(all_subjects) * len(self.recommended_configs)):.1%}")
        
        # Show the single feature combination performance
        print(f"\n📊 FEATURE COMBINATION PERFORMANCE:")
        if combo_summary_data:
            combo = combo_summary_data[0]
            print(f"   Features: {combo['features_list']}")
            print(f"   Performance: {combo['balanced_score_mean']:.4f} ± {combo['balanced_score_std']:.4f}")
            print(f"   Success rate: {combo['success_rate']:.1%} ({combo['n_successful_subjects']}/{len(all_subjects)} subjects)")
        
        # Top performing configs
        print(f"\n⚙️  TOP SVM CONFIGURATIONS:")
        for i, config in enumerate(config_summary_df.head(3).itertuples(), 1):
            print(f"   {i}. {config.config_string}: {config.balanced_score_mean:.4f} ± {config.balanced_score_std:.4f}")
        
        # Subject performance distribution
        print(f"\n👥 SUBJECT PERFORMANCE DISTRIBUTION:")
        successful_subjects = subject_summary_df[subject_summary_df['n_successful_combinations'] > 0]
        print(f"   Subjects with successful models: {len(successful_subjects)}/{len(all_subjects)} ({len(successful_subjects)/len(all_subjects):.1%})")
        if len(successful_subjects) > 0:
            print(f"   Average best score: {successful_subjects['best_balanced_score'].mean():.4f}")
            print(f"   Score range: {successful_subjects['best_balanced_score'].min():.4f} - {successful_subjects['best_balanced_score'].max():.4f}")


def main():
    """Main execution function."""
    print("🎯 Focused Feature Combination Trainer")
    print("Training SVM models using only recommended feature combinations")
    print("Creating comprehensive performance summaries")
    
    trainer = FocusedCombinationTrainer(data_directory='.')
    
    try:
        print(f"\n🔧 CONFIGURATION:")
        print(f"   Recommended features: {', '.join(trainer.recommended_features)}")
        print(f"   Number of features: {len(trainer.recommended_features)}")
        print(f"   SVM configurations: {len(trainer.recommended_configs)}")
        
        print(f"\n   SVM configs to test:")
        for i, config in enumerate(trainer.recommended_configs, 1):
            print(f"     {i}. {config['kernel']} kernel, C={config['C']}, γ={config['gamma']}")
        
        results = trainer.run_focused_training()
        
        if results and results['all_results']:
            all_results = results['all_results']
            best_result = max(all_results, key=lambda x: x['balanced_score_mean'])
            
            print(f"\n🏆 FINAL RESULTS")
            print(f"Total successful experiments: {len(all_results)}")
            print(f"Best performance: {best_result['balanced_score_mean']:.4f}")
            print(f"Best subject: {best_result['subject_id']}")
            print(f"Best feature combination: {best_result['features_list']}")
            print(f"Best SVM configuration: {best_result['config_string']}")
            
            # Performance statistics
            scores = [r['balanced_score_mean'] for r in all_results]
            print(f"\nPerformance statistics:")
            print(f"  Mean: {np.mean(scores):.4f}")
            print(f"  Std:  {np.std(scores):.4f}")
            print(f"  Min:  {np.min(scores):.4f}")
            print(f"  Max:  {np.max(scores):.4f}")
            
            print(f"\n📁 OUTPUT FILES CREATED:")
            print("• focused_detailed_results.csv - All experiment results")
            print("• focused_combination_summary.csv - Performance by feature combination")
            print("• focused_subject_summary.csv - Performance by subject")
            print("• focused_cross_subject_evaluation.csv - Cross-subject test results")
            print("• focused_generalization_summary.csv - Generalization analysis")
            print("• focused_config_summary.csv - Performance by SVM configuration")
            print("• focused_best_model.pkl - Best performing model")
            print("• focused_overall_summary.csv - Overall summary")
            
            print(f"\n🔍 ANALYSIS RECOMMENDATIONS:")
            print("1. Check 'focused_combination_summary.csv' for overall performance of the 6-feature set")
            print("2. Use 'focused_subject_summary.csv' to see which subjects perform best with these features")
            print("3. Review 'focused_generalization_summary.csv' for cross-subject performance")
            print("4. Compare 'focused_config_summary.csv' to validate your recommended SVM parameters")
            print("5. The best model is saved in 'focused_best_model.pkl' for future use")
            
            # Recommendations based on results
            if best_result['balanced_score_mean'] > 0.8:
                print(f"\n✅ EXCELLENT PERFORMANCE! The recommended combinations work very well.")
            elif best_result['balanced_score_mean'] > 0.7:
                print(f"\n✅ GOOD PERFORMANCE! The recommended combinations show promise.")
            elif best_result['balanced_score_mean'] > 0.6:
                print(f"\n⚠️  MODERATE PERFORMANCE. Consider refining feature selection.")
            else:
                print(f"\n❌ LIMITED PERFORMANCE. The recommended combinations may need revision.")
            
            success_rate = len(all_results) / (len(results['subjects']) * len(trainer.recommended_configs))
            if success_rate > 0.8:
                print(f"✅ HIGH SUCCESS RATE ({success_rate:.1%}). Approach is robust.")
            elif success_rate > 0.6:
                print(f"⚠️  MODERATE SUCCESS RATE ({success_rate:.1%}). Some combinations may not work for all subjects.")
            else:
                print(f"❌ LOW SUCCESS RATE ({success_rate:.1%}). Many combinations failed - data quality issues?")
                
        else:
            print("❌ Training failed - no successful experiments")
            print("\n💡 TROUBLESHOOTING:")
            print("• Check if CSV files are in the correct format")
            print("• Verify that the recommended features exist in your data")
            print("• Ensure sufficient samples per subject")
            print("• Check class balance in your data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()