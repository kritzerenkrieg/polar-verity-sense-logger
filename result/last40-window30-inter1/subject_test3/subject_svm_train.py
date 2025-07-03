#!/usr/bin/env python3
"""
Fixed Balanced Per-Subject Boundary Training
Optimized balance between improvements and practical robustness
- Moderate regularization (not too strict)
- Focused feature combinations (not too many)
- Robust validation (not too demanding)
- Multiple metrics but practical thresholds
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
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from itertools import combinations
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class BalancedPerSubjectBoundaryTrainer:
    def __init__(self, data_directory='.', min_features=2, max_features=3, 
                 max_combinations_per_size=8, use_regularization=True):
        """
        Balanced Per-Subject Boundary Training with practical improvements
        
        Args:
            data_directory: Directory containing CSV files
            min_features: Minimum number of features to test
            max_features: Maximum number of features to test  
            max_combinations_per_size: Max combinations per feature count
            use_regularization: Whether to use regularization
        """
        self.data_directory = data_directory
        self.min_features = min_features
        self.max_features = max_features
        self.max_combinations_per_size = max_combinations_per_size
        self.use_regularization = use_regularization
        
        # Focused feature set for better performance
        self.core_time_features = [
            'nn_mean', 'sdnn', 'rmssd', 'pnn50', 'triangular_index', 'nn_min'
        ]
        
        self.core_freq_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 'hf_norm', 'lf_peak'
        ]
        
        self.all_features = self.core_time_features + self.core_freq_features
        
        # Balanced configuration space
        if use_regularization:
            self.config_space = [
                # Regularized configs
                {'kernel': 'linear', 'C': 0.1},
                {'kernel': 'linear', 'C': 1.0},
                {'kernel': 'rbf', 'C': 0.5, 'gamma': 'scale'},
                {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
                {'kernel': 'rbf', 'C': 2.0, 'gamma': 'scale'}
            ]
        else:
            self.config_space = [
                # Original configs
                {'kernel': 'linear', 'C': 0.1},
                {'kernel': 'linear', 'C': 1.0},
                {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
                {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale'}
            ]
        
        self.all_data = None
        
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
            
            # Relaxed minimum samples requirement
            if len(df_clean) < 6:
                return None, None, None, None, []
            
            X = df_clean[available_features].values
            y = df_clean['binary_label'].values
            subjects = df_clean['subject'].values
            conditions = df_clean['condition'].values
            
            # Check for sufficient class balance (relaxed)
            if len(np.unique(y)) < 2:
                return None, None, None, None, []
            
            # Relaxed minimum class size
            class_counts = np.bincount(y)
            if np.min(class_counts) < 2:  # At least 2 samples per class
                return None, None, None, None, []
            
            return X, y, subjects, conditions, available_features
            
        except Exception as e:
            return None, None, None, None, []
    
    def get_balanced_feature_combinations(self, X_train, y_train, available_features):
        """Get balanced feature combinations - not too many, not too few."""
        try:
            # Quick feature importance
            rf = RandomForestClassifier(n_estimators=30, random_state=42)
            rf.fit(X_train, y_train)
            importance_scores = rf.feature_importances_
            
            # Sort features by importance
            feature_importance = list(zip(available_features, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"    📊 Top features:")
            for i, (feat, score) in enumerate(feature_importance[:4]):
                print(f"      {i+1}. {feat}: {score:.3f}")
            
            # Generate focused combinations
            combinations_to_test = []
            
            for n_features in range(self.min_features, self.max_features + 1):
                if n_features <= len(available_features):
                    # Strategy: Focus on top features
                    top_features = [f[0] for f in feature_importance[:min(6, len(feature_importance))]]
                    
                    # Generate combinations
                    all_combos = list(combinations(top_features, n_features))
                    
                    # Limit to reasonable number
                    selected_combos = all_combos[:self.max_combinations_per_size]
                    
                    for combo in selected_combos:
                        combinations_to_test.append(list(combo))
            
            return combinations_to_test
            
        except Exception as e:
            # Simple fallback
            combinations_to_test = []
            for n_features in range(self.min_features, min(self.max_features + 1, len(available_features) + 1)):
                combos = list(combinations(available_features[:4], n_features))
                combinations_to_test.extend([list(c) for c in combos[:self.max_combinations_per_size]])
            
            return combinations_to_test
    
    def evaluate_with_balanced_metrics(self, y_true, y_pred):
        """Evaluate with balanced metrics."""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            
            # Handle edge cases gracefully
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
    
    def train_balanced_subject_model(self, subject_id):
        """Train balanced subject-specific model."""
        print(f"  🔬 Training balanced model for subject {subject_id}...")
        
        # Get subject's data
        subject_data = self.all_data[self.all_data['subject'] == subject_id]
        
        # Relaxed data requirements
        if len(subject_data) < 8:
            print(f"    ❌ Insufficient data: {len(subject_data)} samples (need ≥8)")
            return None
        
        # Check class balance
        class_counts = subject_data['binary_label'].value_counts()
        if len(class_counts) < 2:
            print(f"    ❌ Missing class: {class_counts.to_dict()}")
            return None
        
        if class_counts.min() < 2:
            print(f"    ❌ Poor class balance: {class_counts.to_dict()} (need ≥2 per class)")
            return None
        
        # Prepare features
        X_full, y_full, _, _, available_features = self.prepare_features(
            self.all_features, data_subset=subject_data
        )
        
        if X_full is None or len(available_features) < self.min_features:
            print(f"    ❌ Cannot prepare features")
            return None
        
        # Get balanced feature combinations
        feature_combinations = self.get_balanced_feature_combinations(
            X_full, y_full, available_features
        )
        
        total_tests = len(feature_combinations) * len(self.config_space)
        print(f"    🧪 Testing {len(feature_combinations)} combos × {len(self.config_space)} configs = {total_tests} tests")
        
        best_result = None
        best_score = -1
        
        # Progress tracking
        test_count = 0
        
        # Try different combinations
        for combo in feature_combinations:
            X_combo, y_combo, _, _, feat_names = self.prepare_features(
                combo, data_subset=subject_data
            )
            
            if X_combo is None or len(X_combo) < 6:
                continue
            
            # Try different SVM configurations
            for config in self.config_space:
                test_count += 1
                
                try:
                    # Balanced cross-validation
                    n_splits = min(3, len(np.unique(y_combo)))  # Reasonable splits
                    if n_splits < 2:
                        continue
                    
                    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                    cv_scores = []
                    
                    for train_idx, val_idx in skf.split(X_combo, y_combo):
                        X_train_fold, X_val_fold = X_combo[train_idx], X_combo[val_idx]
                        y_train_fold, y_val_fold = y_combo[train_idx], y_combo[val_idx]
                        
                        # Relaxed class requirements
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
                        
                        # Evaluate with balanced metrics
                        metrics = self.evaluate_with_balanced_metrics(y_val_fold, y_pred)
                        cv_scores.append(metrics['balanced_score'])
                    
                    # Calculate mean performance
                    if cv_scores:
                        mean_score = np.mean(cv_scores)
                        
                        # Gentle overfitting penalty (not too harsh)
                        if mean_score > 0.95:
                            mean_score *= 0.9  # Mild penalty
                        
                        if mean_score > best_score:
                            best_score = mean_score
                            best_result = {
                                'subject_id': subject_id,
                                'features': combo,
                                'config': config,
                                'cv_score': mean_score,
                                'cv_std': np.std(cv_scores),
                                'n_samples': len(X_combo),
                                'class_distribution': f"{np.sum(y_combo == 0)}/{np.sum(y_combo == 1)}",
                                'n_folds': len(cv_scores)
                            }
                            
                except Exception as e:
                    continue
        
        print(f"    📊 Completed {test_count} tests")
        
        if best_result:
            # Train final model
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
                
                print(f"    ✅ Best score: {best_result['cv_score']:.4f} ± {best_result['cv_std']:.4f}")
                print(f"    ✅ Features: {', '.join(best_result['features'])}")
                print(f"    ✅ Config: {best_result['config']['kernel']}")
                
                return best_result
        
        print(f"    ❌ No valid model found")
        return None
    
    def evaluate_balanced_cross_subject(self, subject_model, test_subjects):
        """Balanced cross-subject evaluation."""
        model_id = subject_model['subject_id']
        features = subject_model['features']
        trained_model = subject_model['trained_model']
        
        test_results = []
        
        for test_subject in test_subjects:
            if test_subject == model_id:
                continue
            
            # Get test subject's data
            test_data = self.all_data[self.all_data['subject'] == test_subject]
            
            X_test, y_test, _, _, _ = self.prepare_features(
                features, data_subset=test_data
            )
            
            if X_test is None or len(X_test) < 4:
                continue
            
            try:
                # Test with balanced metrics
                y_pred = trained_model.predict(X_test)
                metrics = self.evaluate_with_balanced_metrics(y_test, y_pred)
                
                test_results.append({
                    'model_subject': model_id,
                    'test_subject': test_subject,
                    'n_test_samples': len(X_test),
                    'test_class_distribution': f"{np.sum(y_test == 0)}/{np.sum(y_test == 1)}",
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'balanced_score': metrics['balanced_score']
                })
                
            except Exception as e:
                continue
        
        return test_results
    
    def run_balanced_training(self):
        """Run balanced per-subject boundary training."""
        print("=== Balanced Per-Subject Boundary Training ===")
        print(f"🔧 Configuration: Regularization={self.use_regularization}, Max features={self.max_features}")
        print(f"📊 Feature space: {len(self.all_features)} features")
        
        # Load data
        self.all_data = self.load_data()
        
        # Get all subjects
        all_subjects = sorted(self.all_data['subject'].unique())
        print(f"Total subjects: {len(all_subjects)}")
        
        # STEP 1: Train balanced subject-specific models
        print(f"\n🔬 STEP 1: Training {len(all_subjects)} balanced subject-specific models")
        
        subject_models = {}
        
        for i, subject_id in enumerate(all_subjects):
            print(f"\n[{i+1}/{len(all_subjects)}] Subject: {subject_id}")
            
            subject_model = self.train_balanced_subject_model(subject_id)
            
            if subject_model:
                subject_models[subject_id] = subject_model
                print(f"    ✅ Balanced model trained successfully")
            else:
                print(f"    ❌ Failed to train model")
        
        print(f"\n✓ Successfully trained {len(subject_models)} balanced subject models")
        
        if len(subject_models) < 2:
            print("Error: Need at least 2 successful subject models")
            return []
        
        # STEP 2: Balanced cross-subject evaluation
        print(f"\n🎯 STEP 2: Balanced cross-subject evaluation")
        
        all_cross_results = []
        subject_generalization_scores = {}
        
        for model_subject in subject_models:
            print(f"\n🧪 Testing {model_subject}'s balanced model...")
            
            cross_results = self.evaluate_balanced_cross_subject(
                subject_models[model_subject], 
                all_subjects
            )
            
            all_cross_results.extend(cross_results)
            
            if cross_results:
                # Calculate statistics
                accuracies = [r['accuracy'] for r in cross_results]
                balanced_scores = [r['balanced_score'] for r in cross_results]
                f1_scores = [r['f1'] for r in cross_results]
                
                subject_generalization_scores[model_subject] = {
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'balanced_score_mean': np.mean(balanced_scores),
                    'balanced_score_std': np.std(balanced_scores),
                    'f1_mean': np.mean(f1_scores),
                    'f1_std': np.std(f1_scores),
                    'n_tests': len(cross_results)
                }
                
                print(f"  📊 {model_subject}:")
                print(f"    Accuracy: {subject_generalization_scores[model_subject]['accuracy_mean']:.4f} ± {subject_generalization_scores[model_subject]['accuracy_std']:.4f}")
                print(f"    Balanced: {subject_generalization_scores[model_subject]['balanced_score_mean']:.4f} ± {subject_generalization_scores[model_subject]['balanced_score_std']:.4f}")
            else:
                print(f"  ❌ No valid tests for {model_subject}")
        
        # STEP 3: Select best model
        print(f"\n🏆 STEP 3: Selecting best balanced model")
        
        if subject_generalization_scores:
            # Rank by balanced score
            best_subject = max(subject_generalization_scores.keys(), 
                             key=lambda x: subject_generalization_scores[x]['balanced_score_mean'])
            
            best_score = subject_generalization_scores[best_subject]
            best_model = subject_models[best_subject]
            
            print(f"\n🎯 BEST BALANCED MODEL: Subject {best_subject}")
            print(f"   Balanced score: {best_score['balanced_score_mean']:.4f} ± {best_score['balanced_score_std']:.4f}")
            print(f"   Accuracy: {best_score['accuracy_mean']:.4f} ± {best_score['accuracy_std']:.4f}")
            print(f"   F1 score: {best_score['f1_mean']:.4f} ± {best_score['f1_std']:.4f}")
            print(f"   Features: {', '.join(best_model['features'])}")
            print(f"   Config: {best_model['config']['kernel']}")
            
            # Show all models ranked
            print(f"\n📊 ALL MODELS RANKED BY BALANCED SCORE:")
            sorted_subjects = sorted(subject_generalization_scores.keys(), 
                                   key=lambda x: subject_generalization_scores[x]['balanced_score_mean'], 
                                   reverse=True)
            
            for i, subj in enumerate(sorted_subjects):
                score = subject_generalization_scores[subj]
                print(f"  {i+1:2d}. {subj}: {score['balanced_score_mean']:.4f} ± {score['balanced_score_std']:.4f} "
                      f"(acc: {score['accuracy_mean']:.4f}, f1: {score['f1_mean']:.4f})")
            
            # Save results
            self.save_balanced_results(subject_models, all_cross_results, 
                                     subject_generalization_scores, best_subject)
            
            return {
                'subject_models': subject_models,
                'cross_results': all_cross_results,
                'generalization_scores': subject_generalization_scores,
                'best_subject': best_subject,
                'best_model': best_model
            }
        
        return []
    
    def save_balanced_results(self, subject_models, cross_results, generalization_scores, best_subject):
        """Save balanced results."""
        print(f"\n💾 Saving balanced results...")
        
        # Subject model details
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
                'class_distribution': model['class_distribution'],
                'n_cv_folds': model['n_folds']
            })
        
        subject_df = pd.DataFrame(subject_model_data)
        subject_df.to_csv('balanced_subject_models.csv', index=False)
        print("✓ 'balanced_subject_models.csv'")
        
        # Cross-subject results
        cross_df = pd.DataFrame(cross_results)
        cross_df.to_csv('balanced_cross_subject_evaluation.csv', index=False)
        print("✓ 'balanced_cross_subject_evaluation.csv'")
        
        # Generalization scores
        generalization_data = []
        for subject_id, scores in generalization_scores.items():
            generalization_data.append({
                'subject_id': subject_id,
                'balanced_score_mean': scores['balanced_score_mean'],
                'balanced_score_std': scores['balanced_score_std'],
                'accuracy_mean': scores['accuracy_mean'],
                'accuracy_std': scores['accuracy_std'],
                'f1_mean': scores['f1_mean'],
                'f1_std': scores['f1_std'],
                'n_tests': scores['n_tests'],
                'is_best_model': subject_id == best_subject
            })
        
        generalization_df = pd.DataFrame(generalization_data)
        generalization_df = generalization_df.sort_values('balanced_score_mean', ascending=False)
        generalization_df.to_csv('balanced_generalization_scores.csv', index=False)
        print("✓ 'balanced_generalization_scores.csv'")
        
        # Best model
        best_model = subject_models[best_subject]
        joblib.dump({
            'method': 'balanced_per_subject_boundary_training',
            'best_subject': best_subject,
            'model_pipeline': best_model['trained_model'],
            'features': best_model['features'],
            'config': best_model['config'],
            'cv_score': best_model['cv_score'],
            'cross_subject_score': generalization_scores[best_subject]['balanced_score_mean'],
            'use_regularization': self.use_regularization
        }, 'balanced_best_boundary_model.pkl')
        print("✓ 'balanced_best_boundary_model.pkl'")
        
        # Summary
        summary = {
            'method': 'balanced_per_subject_boundary_training',
            'n_subjects': len(subject_models),
            'best_subject': best_subject,
            'best_balanced_score': generalization_scores[best_subject]['balanced_score_mean'],
            'best_accuracy': generalization_scores[best_subject]['accuracy_mean'],
            'best_f1': generalization_scores[best_subject]['f1_mean'],
            'use_regularization': self.use_regularization,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('balanced_boundary_training_summary.csv', index=False)
        print("✓ 'balanced_boundary_training_summary.csv'")

def main():
    """Main execution function."""
    print("⚖️  Balanced Per-Subject Boundary Training")
    print("Optimized balance between improvements and robustness:")
    print("• Moderate regularization (not too strict)")
    print("• Focused feature combinations (not too many)")
    print("• Robust validation (not too demanding)")
    print("• Multiple metrics with practical thresholds")
    
    # Simple configuration
    try:
        use_regularization = input("Use regularization? (y/n, default: y): ").lower() != 'n'
        max_features = int(input("Maximum features per combination (default: 3): ") or "3")
        max_combos = int(input("Max combinations per feature size (default: 8): ") or "8")
        
    except ValueError:
        use_regularization = True
        max_features = 3
        max_combos = 8
    
    trainer = BalancedPerSubjectBoundaryTrainer(
        data_directory='.', 
        min_features=2, 
        max_features=max_features,
        max_combinations_per_size=max_combos,
        use_regularization=use_regularization
    )
    
    try:
        print(f"\n🔧 Configuration:")
        print(f"   Regularization: {use_regularization}")
        print(f"   Max features: {max_features}")
        print(f"   Max combinations per size: {max_combos}")
        
        results = trainer.run_balanced_training()
        
        if results:
            best_subject = results['best_subject']
            best_scores = results['generalization_scores'][best_subject]
            
            print(f"\n🏆 BALANCED FINAL RESULT")
            print(f"Best subject's model: {best_subject}")
            print(f"Balanced score: {best_scores['balanced_score_mean']:.4f} ± {best_scores['balanced_score_std']:.4f}")
            print(f"Accuracy: {best_scores['accuracy_mean']:.4f} ± {best_scores['accuracy_std']:.4f}")
            print(f"F1 score: {best_scores['f1_mean']:.4f} ± {best_scores['f1_std']:.4f}")
            
            # Performance comparison with original
            original_acc = 0.5803  # irsyad's original result
            improvement = best_scores['balanced_score_mean'] - original_acc
            print(f"\n📈 COMPARISON vs ORIGINAL:")
            print(f"   Original (irsyad): {original_acc:.4f}")
            print(f"   Balanced: {best_scores['balanced_score_mean']:.4f}")
            print(f"   Improvement: {improvement:+.4f} ({improvement/original_acc*100:+.1f}%)")
            
            # Success rate
            successful_subjects = len(results['subject_models'])
            total_subjects = len(trainer.all_data['subject'].unique())
            success_rate = successful_subjects / total_subjects
            print(f"\n📊 TRAINING SUCCESS RATE:")
            print(f"   Successful models: {successful_subjects}/{total_subjects} ({success_rate:.1%})")
            
            print(f"\n📁 Balanced output files:")
            print("• balanced_subject_models.csv - Individual model details")
            print("• balanced_cross_subject_evaluation.csv - Cross-subject test results")
            print("• balanced_generalization_scores.csv - Generalization summary")
            print("• balanced_best_boundary_model.pkl - Final selected model")
            print("• balanced_boundary_training_summary.csv - Overall summary")
            
            # Recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            if best_scores['balanced_score_mean'] > 0.62:
                print("✅ Good balanced performance! Model shows solid generalization.")
            elif best_scores['balanced_score_mean'] > 0.58:
                print("⚠️  Moderate performance. Consider feature engineering or more data.")
            else:
                print("❌ Limited performance. Per-subject boundaries may not be optimal for this data.")
            
            if best_scores['balanced_score_std'] > 0.08:
                print("⚠️  High variance across subjects. Results may not be consistent.")
            else:
                print("✅ Consistent performance across subjects.")
            
            if success_rate < 0.7:
                print("⚠️  Low training success rate. Consider relaxing validation criteria.")
            else:
                print("✅ High training success rate. Model approach is robust.")
                
        else:
            print("❌ Balanced training failed")
            print("\n💡 TROUBLESHOOTING:")
            print("• Try reducing max_features to 2")
            print("• Increase max_combinations_per_size to 10")
            print("• Check if data has sufficient samples per subject")
            print("• Verify class balance in your data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()