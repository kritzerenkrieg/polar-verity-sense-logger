#!/usr/bin/env python3
"""
Per-Subject Boundary Training + Meta-Selection
Following professor's exact requirements:
1. Train a model on the boundary of each subject
2. Find the best model that suits all subjects
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
import joblib
import warnings
from itertools import combinations
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class PerSubjectBoundaryTrainer:
    def __init__(self, data_directory='.', min_features=2, max_features=3, 
                 max_combinations_per_size=8):
        """
        Per-Subject Boundary Training + Meta-Selection
        
        Args:
            data_directory: Directory containing CSV files
            min_features: Minimum number of features to test
            max_features: Maximum number of features to test
            max_combinations_per_size: Max combinations per feature count
        """
        self.data_directory = data_directory
        self.min_features = min_features
        self.max_features = max_features
        self.max_combinations_per_size = max_combinations_per_size
        
        # Core features for efficiency
        self.core_time_features = [
            'nn_mean', 'sdnn', 'rmssd', 'pnn50', 'triangular_index'
        ]
        
        self.core_freq_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 'hf_norm'
        ]
        
        self.all_features = self.core_time_features + self.core_freq_features
        
        # SVM configurations to test
        self.config_space = [
            {'kernel': 'linear', 'C': 0.1},
            {'kernel': 'linear', 'C': 1.0},
            {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}
        ]
        
        self.all_data = None
        self.subject_models = {}  # Store per-subject models
        
    def load_data(self):
        """Load and combine all CSV files."""
        print("Loading CSV files...")
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
        
        return combined_df
    
    def prepare_features(self, feature_list, data_subset=None):
        """Prepare feature matrix for a specific feature combination."""
        if data_subset is None:
            data_subset = self.all_data
            
        try:
            time_df = data_subset[data_subset['domain'] == 'time'].copy()
            freq_df = data_subset[data_subset['domain'] == 'frequency'].copy()
            
            merge_cols = ['subject', 'condition', 'label', 'binary_label']
            
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
            
            if len(np.unique(y)) < 2:
                return None, None, None, None, []
            
            return X, y, subjects, conditions, available_features
            
        except Exception as e:
            return None, None, None, None, []
    
    def get_smart_feature_combinations(self, X_train, y_train, available_features):
        """Get smart feature combinations based on importance."""
        try:
            # Quick feature importance
            rf = RandomForestClassifier(n_estimators=20, random_state=42)
            rf.fit(X_train, y_train)
            importance_scores = rf.feature_importances_
            
            # Sort features by importance
            feature_importance = list(zip(available_features, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Generate combinations
            combinations_to_test = []
            
            for n_features in range(self.min_features, self.max_features + 1):
                if n_features <= len(available_features):
                    # Take top features
                    top_features = [f[0] for f in feature_importance[:min(6, len(feature_importance))]]
                    
                    # Generate combinations
                    all_combos = list(combinations(top_features, n_features))
                    
                    # Limit combinations
                    selected_combos = all_combos[:self.max_combinations_per_size]
                    
                    for combo in selected_combos:
                        combinations_to_test.append(list(combo))
            
            return combinations_to_test
            
        except Exception as e:
            # Fallback
            combinations_to_test = []
            for n_features in range(self.min_features, min(self.max_features + 1, len(available_features) + 1)):
                combos = list(combinations(available_features[:5], n_features))
                combinations_to_test.extend([list(c) for c in combos[:self.max_combinations_per_size]])
            
            return combinations_to_test
    
    def train_subject_specific_model(self, subject_id):
        """
        STEP 1: Train a model on the boundary of ONE subject
        This learns the decision boundary specific to this subject's data
        """
        print(f"  Training model for subject {subject_id}...")
        
        # Get ONLY this subject's data
        subject_data = self.all_data[self.all_data['subject'] == subject_id]
        
        if len(subject_data) < 10:
            print(f"    ❌ Insufficient data for subject {subject_id}")
            return None
        
        # Prepare features for this subject
        X_full, y_full, _, _, available_features = self.prepare_features(
            self.all_features, data_subset=subject_data
        )
        
        if X_full is None or len(available_features) < self.min_features:
            print(f"    ❌ Cannot prepare features for subject {subject_id}")
            return None
        
        # Get smart feature combinations
        feature_combinations = self.get_smart_feature_combinations(X_full, y_full, available_features)
        
        print(f"    Testing {len(feature_combinations)} feature combinations × {len(self.config_space)} configs")
        
        best_result = None
        best_score = -1
        
        # Try different feature combinations and SVM configs
        for combo in feature_combinations:
            X_combo, y_combo, _, _, feat_names = self.prepare_features(
                combo, data_subset=subject_data
            )
            
            if X_combo is None or len(X_combo) < 6:
                continue
            
            # Try different SVM configurations
            for config in self.config_space:
                try:
                    # Use stratified CV on this subject's data
                    n_splits = min(3, len(np.unique(y_combo)))
                    if n_splits < 2:
                        continue
                    
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    scores = []
                    
                    for train_idx, val_idx in skf.split(X_combo, y_combo):
                        X_train_fold, X_val_fold = X_combo[train_idx], X_combo[val_idx]
                        y_train_fold, y_val_fold = y_combo[train_idx], y_combo[val_idx]
                        
                        if len(np.unique(y_train_fold)) < 2:
                            continue
                        
                        # Scale and train
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_fold)
                        X_val_scaled = scaler.transform(X_val_fold)
                        
                        svm = SVC(**config, random_state=42)
                        svm.fit(X_train_scaled, y_train_fold)
                        
                        y_pred = svm.predict(X_val_scaled)
                        scores.append(accuracy_score(y_val_fold, y_pred))
                    
                    if scores:
                        mean_score = np.mean(scores)
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_result = {
                                'subject_id': subject_id,
                                'features': combo,
                                'config': config,
                                'cv_score': mean_score,
                                'cv_std': np.std(scores),
                                'n_samples': len(X_combo),
                                'class_distribution': f"{np.sum(y_combo == 0)}/{np.sum(y_combo == 1)}"
                            }
                            
                except Exception as e:
                    continue
        
        if best_result:
            # Train final model on all subject's data
            X_final, y_final, _, _, _ = self.prepare_features(
                best_result['features'], data_subset=subject_data
            )
            
            if X_final is not None:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(**best_result['config'], random_state=42, probability=True))
                ])
                
                pipeline.fit(X_final, y_final)
                
                best_result['trained_model'] = pipeline
                print(f"    ✓ Best CV score: {best_result['cv_score']:.4f}")
                print(f"    ✓ Features: {', '.join(best_result['features'])}")
                print(f"    ✓ Config: {best_result['config']['kernel']}")
                
                return best_result
        
        print(f"    ❌ No valid model found for subject {subject_id}")
        return None
    
    def evaluate_model_on_other_subjects(self, subject_model, test_subjects):
        """
        STEP 2: Test this subject's model on ALL other subjects
        This shows how well this subject's boundary generalizes
        """
        model_id = subject_model['subject_id']
        features = subject_model['features']
        trained_model = subject_model['trained_model']
        
        test_results = []
        
        for test_subject in test_subjects:
            if test_subject == model_id:
                continue  # Skip self
            
            # Get test subject's data
            test_data = self.all_data[self.all_data['subject'] == test_subject]
            
            X_test, y_test, _, _, _ = self.prepare_features(
                features, data_subset=test_data
            )
            
            if X_test is None or len(X_test) < 4:
                continue
            
            try:
                # Test the subject-specific model on this test subject
                y_pred = trained_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                test_results.append({
                    'model_subject': model_id,
                    'test_subject': test_subject,
                    'accuracy': accuracy,
                    'n_test_samples': len(X_test),
                    'test_class_distribution': f"{np.sum(y_test == 0)}/{np.sum(y_test == 1)}"
                })
                
            except Exception as e:
                continue
        
        return test_results
    
    def run_per_subject_boundary_training(self):
        """
        Main method implementing professor's requirements:
        1. Train a model on the boundary of each subject
        2. Find the best model that suits all subjects
        """
        print("=== Per-Subject Boundary Training + Meta-Selection ===")
        print("Step 1: Train individual models on each subject's boundary")
        print("Step 2: Find which subject's model generalizes best to all others")
        
        # Load data
        self.all_data = self.load_data()
        
        # Get all subjects
        all_subjects = sorted(self.all_data['subject'].unique())
        print(f"Total subjects: {len(all_subjects)}")
        
        if len(all_subjects) < 3:
            print("Error: Need at least 3 subjects")
            return []
        
        # STEP 1: Train one model per subject
        print(f"\n🔬 STEP 1: Training {len(all_subjects)} subject-specific models")
        
        subject_models = {}
        
        for i, subject_id in enumerate(all_subjects):
            print(f"\n[{i+1}/{len(all_subjects)}] Subject: {subject_id}")
            
            subject_model = self.train_subject_specific_model(subject_id)
            
            if subject_model:
                subject_models[subject_id] = subject_model
                print(f"    ✅ Model trained successfully")
            else:
                print(f"    ❌ Failed to train model")
        
        print(f"\n✓ Successfully trained {len(subject_models)} subject-specific models")
        
        if len(subject_models) < 2:
            print("Error: Need at least 2 successful subject models")
            return []
        
        # STEP 2: Test each subject's model on all other subjects
        print(f"\n🎯 STEP 2: Testing each model on all other subjects")
        
        all_cross_subject_results = []
        subject_generalization_scores = {}
        
        for model_subject in subject_models:
            print(f"\nTesting {model_subject}'s model on other subjects...")
            
            cross_results = self.evaluate_model_on_other_subjects(
                subject_models[model_subject], 
                all_subjects
            )
            
            all_cross_subject_results.extend(cross_results)
            
            if cross_results:
                accuracies = [r['accuracy'] for r in cross_results]
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                
                subject_generalization_scores[model_subject] = {
                    'mean_accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy,
                    'n_tests': len(cross_results),
                    'individual_accuracies': accuracies
                }
                
                print(f"  {model_subject}: {mean_accuracy:.4f} ± {std_accuracy:.4f} (tested on {len(cross_results)} subjects)")
            else:
                print(f"  {model_subject}: No valid tests")
        
        # STEP 3: Find the best generalizing model
        print(f"\n🏆 STEP 3: Selecting best generalizing model")
        
        if subject_generalization_scores:
            # Sort by mean accuracy
            best_subject = max(subject_generalization_scores.keys(), 
                             key=lambda x: subject_generalization_scores[x]['mean_accuracy'])
            
            best_score = subject_generalization_scores[best_subject]
            best_model = subject_models[best_subject]
            
            print(f"\n🎯 BEST MODEL: Subject {best_subject}")
            print(f"   Cross-subject accuracy: {best_score['mean_accuracy']:.4f} ± {best_score['std_accuracy']:.4f}")
            print(f"   Features: {', '.join(best_model['features'])}")
            print(f"   Config: {best_model['config']['kernel']} {best_model['config']}")
            print(f"   Original CV score: {best_model['cv_score']:.4f}")
            
            # Show all models ranked
            print(f"\n📊 ALL MODELS RANKED BY GENERALIZATION:")
            sorted_subjects = sorted(subject_generalization_scores.keys(), 
                                   key=lambda x: subject_generalization_scores[x]['mean_accuracy'], 
                                   reverse=True)
            
            for i, subj in enumerate(sorted_subjects):
                score = subject_generalization_scores[subj]
                print(f"  {i+1}. {subj}: {score['mean_accuracy']:.4f} ± {score['std_accuracy']:.4f} "
                      f"(tested on {score['n_tests']} subjects)")
            
            # Save results
            self.save_results(subject_models, all_cross_subject_results, 
                            subject_generalization_scores, best_subject)
            
            return {
                'subject_models': subject_models,
                'cross_subject_results': all_cross_subject_results,
                'generalization_scores': subject_generalization_scores,
                'best_model_subject': best_subject,
                'best_model': best_model
            }
        
        else:
            print("❌ No valid generalization scores computed")
            return []
    
    def save_results(self, subject_models, cross_results, generalization_scores, best_subject):
        """Save per-subject boundary training results."""
        print(f"\n💾 Saving results...")
        
        # 1. Subject-specific model details
        subject_model_data = []
        for subject_id, model in subject_models.items():
            subject_model_data.append({
                'subject_id': subject_id,
                'features': ', '.join(model['features']),
                'n_features': len(model['features']),
                'kernel': model['config']['kernel'],
                'config_params': str({k: v for k, v in model['config'].items() if k != 'kernel'}),
                'cv_score': model['cv_score'],
                'cv_std': model['cv_std'],
                'n_samples': model['n_samples'],
                'class_distribution': model['class_distribution']
            })
        
        subject_df = pd.DataFrame(subject_model_data)
        subject_df.to_csv('subject_specific_models.csv', index=False)
        print("✓ 'subject_specific_models.csv'")
        
        # 2. Cross-subject evaluation results
        cross_df = pd.DataFrame(cross_results)
        cross_df.to_csv('cross_subject_evaluation.csv', index=False)
        print("✓ 'cross_subject_evaluation.csv'")
        
        # 3. Generalization scores summary
        generalization_data = []
        for subject_id, scores in generalization_scores.items():
            generalization_data.append({
                'subject_id': subject_id,
                'mean_cross_accuracy': scores['mean_accuracy'],
                'std_cross_accuracy': scores['std_accuracy'],
                'n_tests': scores['n_tests'],
                'individual_accuracies': ', '.join([f'{acc:.3f}' for acc in scores['individual_accuracies']]),
                'is_best_model': subject_id == best_subject
            })
        
        generalization_df = pd.DataFrame(generalization_data)
        generalization_df = generalization_df.sort_values('mean_cross_accuracy', ascending=False)
        generalization_df.to_csv('generalization_scores.csv', index=False)
        print("✓ 'generalization_scores.csv'")
        
        # 4. Best model
        best_model = subject_models[best_subject]
        joblib.dump({
            'method': 'per_subject_boundary_training',
            'best_subject': best_subject,
            'model_pipeline': best_model['trained_model'],
            'features': best_model['features'],
            'config': best_model['config'],
            'cv_score': best_model['cv_score'],
            'cross_subject_score': generalization_scores[best_subject]['mean_accuracy'],
            'cross_subject_std': generalization_scores[best_subject]['std_accuracy']
        }, 'best_subject_boundary_model.pkl')
        print("✓ 'best_subject_boundary_model.pkl'")
        
        # 5. Summary
        summary = {
            'method': 'per_subject_boundary_training',
            'n_subjects': len(subject_models),
            'best_subject': best_subject,
            'best_cross_accuracy': generalization_scores[best_subject]['mean_accuracy'],
            'best_cross_std': generalization_scores[best_subject]['std_accuracy'],
            'timestamp': datetime.now().isoformat()
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('boundary_training_summary.csv', index=False)
        print("✓ 'boundary_training_summary.csv'")

def main():
    """Main execution function."""
    print("🎯 Per-Subject Boundary Training + Meta-Selection")
    print("Following professor's exact requirements:")
    print("1. Train a model on the boundary of each subject")
    print("2. Find the best model that suits all subjects")
    
    trainer = PerSubjectBoundaryTrainer(
        data_directory='.', 
        min_features=2, 
        max_features=6,
        max_combinations_per_size=16
    )
    
    try:
        results = trainer.run_per_subject_boundary_training()
        
        if results:
            best_subject = results['best_model_subject']
            best_score = results['generalization_scores'][best_subject]
            
            print(f"\n🏆 FINAL RESULT")
            print(f"Best subject's model: {best_subject}")
            print(f"Cross-subject accuracy: {best_score['mean_accuracy']:.4f} ± {best_score['std_accuracy']:.4f}")
            print(f"This model generalizes best to other subjects!")
            
            print(f"\n📁 Output files:")
            print("• subject_specific_models.csv - Individual model details")
            print("• cross_subject_evaluation.csv - All cross-subject tests")
            print("• generalization_scores.csv - Summary of generalization")
            print("• best_subject_boundary_model.pkl - Final selected model")
            print("• boundary_training_summary.csv - Overall summary")
                
        else:
            print("❌ Training failed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()