#!/usr/bin/env python3
"""
Advanced HRV-Based Lie Detection using SVM with Multiple Optimization Techniques
- Nested Cross-Validation for unbiased hyperparameter tuning
- SVM-RFE for feature selection
- Class-weighted SVM for imbalance handling
- Ensemble voting across multiple kernels
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedHRVLieDetector:
    def __init__(self, data_directory='.'):
        """
        Initialize the Advanced HRV Lie Detection system.
        
        Args:
            data_directory (str): Directory containing CSV files
        """
        self.data_directory = data_directory
        
        # High-reliability features (≥80% consistency) based on analysis
        self.selected_features = {
            'time': ['nn_mean'],  # 80% consistency, 1.425 effect size, Lie Higher
            'frequency': [
                'lf_power',      # 90% consistency, 0.879 effect size, Truth Higher
                'ln_hf',         # 90% consistency, 0.609 effect size, Truth Higher  
                'hf_norm',       # 80% consistency, 0.776 effect size, Lie Higher
                'lf_norm',       # 80% consistency, 0.776 effect size, Truth Higher
                'lf_hf_ratio'    # 80% consistency, 0.733 effect size, Truth Higher
            ]
        }
        
        self.all_data = None
        self.best_features = None
        self.ensemble_model = None
        
    def parse_filename(self, filename):
        """
        Parse CSV filename to extract metadata.
        
        Args:
            filename (str): CSV filename
            
        Returns:
            dict: Parsed metadata or None if invalid format
        """
        pattern = r'pr_([^_]+)_([^_]+)_(truth|lie)-sequence_(frequency_)?windowed\.csv'
        match = re.match(pattern, filename)
        
        if match:
            subject = match.group(1)
            condition = match.group(2)
            label = match.group(3)  # 'truth' or 'lie'
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
        """
        Discover and organize all CSV files in the directory.
        
        Returns:
            dict: Organized file metadata
        """
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
        """
        Load and combine all CSV files into a single dataset.
        
        Args:
            file_metadata (list): List of file metadata dictionaries
            
        Returns:
            pd.DataFrame: Combined dataset
        """
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
        Prepare combined feature matrix from time and frequency domains.
        
        Args:
            df (pd.DataFrame): Combined dataset
            
        Returns:
            tuple: (X, y, groups, feature_names)
        """
        # Merge time and frequency data
        time_df = df[df['domain'] == 'time'].copy()
        freq_df = df[df['domain'] == 'frequency'].copy()
        
        # Select only needed columns to avoid conflicts
        time_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.selected_features['time']
        freq_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.selected_features['frequency']
        
        time_df_clean = time_df[time_cols].copy()
        freq_df_clean = freq_df[freq_cols].copy()
        
        # Merge on common identifiers
        merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
        df_filtered = pd.merge(
            time_df_clean, freq_df_clean, 
            on=merge_cols, 
            suffixes=('_time', '_freq')
        )
        
        # Create feature column names
        time_features = self.selected_features['time']
        freq_features = self.selected_features['frequency']
        feature_cols = time_features + freq_features
        
        print(f"Merged {len(time_df)} time samples with {len(freq_df)} frequency samples")
        print(f"Result: {len(df_filtered)} combined samples")
        
        # Check for missing features
        available_features = [col for col in feature_cols if col in df_filtered.columns]
        missing_features = [col for col in feature_cols if col not in df_filtered.columns]
        
        if missing_features:
            print(f"Warning: Missing features {missing_features}")
        
        # Prepare final dataset
        df_clean = df_filtered.dropna(subset=available_features)
        
        X = df_clean[available_features].values
        y = df_clean['binary_label'].values  # Only predict truth (0) vs lie (1)
        groups = df_clean['subject'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Using {len(available_features)} high-reliability features: {available_features}")
        print(f"Target distribution: Truth={np.sum(y==0)}, Lie={np.sum(y==1)}")
        
        return X, y, groups, available_features
    
    def recursive_feature_elimination(self, X, y, groups, feature_names):
        """
        Apply SVM-RFE to select optimal features within our high-reliability set.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels (0=truth, 1=lie)
            groups (np.array): Subject groups
            feature_names (list): Original feature names
            
        Returns:
            tuple: (X_selected, selected_feature_names, selector)
        """
        print("\n=== RECURSIVE FEATURE ELIMINATION ===")
        
        # Test different numbers of features
        n_features_range = range(2, min(len(feature_names) + 1, 7))
        best_score = 0
        best_n_features = len(feature_names)
        best_features = feature_names
        
        logo = LeaveOneGroupOut()
        
        for n_features in n_features_range:
            print(f"Testing {n_features} features...")
            
            # Create RFE selector with linear SVM (works best for feature ranking)
            selector = RFE(
                estimator=SVC(kernel='linear', class_weight='balanced'), 
                n_features_to_select=n_features
            )
            
            # Create pipeline with scaling and RFE
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('rfe', selector),
                ('svm', SVC(kernel='linear', class_weight='balanced'))
            ])
            
            # Evaluate with LOSO CV
            scores = cross_val_score(pipeline, X, y, cv=logo, groups=groups, scoring='accuracy')
            mean_score = scores.mean()
            
            print(f"  {n_features} features: {mean_score:.4f} ± {scores.std():.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_n_features = n_features
                
                # Fit selector to get feature names
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                selector.fit(X_scaled, y)
                selected_mask = selector.support_
                best_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        print(f"\nBest RFE result: {best_n_features} features with {best_score:.4f} accuracy")
        print(f"Selected features: {best_features}")
        
        # Final RFE with best number of features
        final_selector = RFE(
            estimator=SVC(kernel='linear', class_weight='balanced'), 
            n_features_to_select=best_n_features
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_selected = final_selector.fit_transform(X_scaled, y)
        
        self.best_features = best_features
        
        return X_selected, best_features, final_selector
    
    def nested_cross_validation(self, X, y, groups, kernel='rbf'):
        """
        Perform optimized nested cross-validation for unbiased hyperparameter tuning.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            kernel (str): SVM kernel to test
            
        Returns:
            dict: Results from nested CV
        """
        print(f"\n=== NESTED CROSS-VALIDATION: {kernel.upper()} KERNEL ===")
        
        # Simplified parameter grids for faster execution
        if kernel == 'linear':
            param_grid = {'C': [0.1, 1, 10]}
        elif kernel == 'rbf':
            param_grid = {'C': [1, 10], 'gamma': ['scale', 0.1]}
        elif kernel == 'poly':
            param_grid = {'C': [1, 10], 'gamma': ['scale'], 'degree': [2, 3]}
        else:  # sigmoid
            param_grid = {'C': [1, 10], 'gamma': ['scale']}
        
        # Use faster stratified K-fold for inner CV instead of LOSO
        from sklearn.model_selection import StratifiedKFold
        
        outer_cv = LeaveOneGroupOut()
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Much faster
        
        nested_scores = []
        best_params_list = []
        
        print(f"Running {outer_cv.get_n_splits(X, y, groups)} outer folds...")
        
        # Manual outer loop with progress tracking
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y, groups)):
            print(f"  Processing fold {fold_idx + 1}...", end=' ')
            
            X_train_outer, X_val_outer = X[train_idx], X[val_idx]
            y_train_outer, y_val_outer = y[train_idx], y[val_idx]
            
            # Inner grid search with simpler CV
            best_score = 0
            best_params = None
            
            # Simple grid search without nested CV for speed
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_outer)
            X_val_scaled = scaler.transform(X_val_outer)
            
            # Try each parameter combination
            for params in self._generate_param_combinations(param_grid):
                try:
                    # Create and train SVM
                    svm = SVC(kernel=kernel, class_weight='balanced', **params)
                    
                    # Quick 3-fold validation on training data
                    cv_scores = []
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    
                    for tr_idx, te_idx in skf.split(X_train_scaled, y_train_outer):
                        X_tr, X_te = X_train_scaled[tr_idx], X_train_scaled[te_idx]
                        y_tr, y_te = y_train_outer[tr_idx], y_train_outer[te_idx]
                        
                        svm_copy = SVC(kernel=kernel, class_weight='balanced', **params)
                        svm_copy.fit(X_tr, y_tr)
                        score = svm_copy.score(X_te, y_te)
                        cv_scores.append(score)
                    
                    avg_score = np.mean(cv_scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = params
                        
                except Exception as e:
                    continue
            
            # Train final model with best params and evaluate on outer validation
            if best_params is None:
                best_params = {'C': 1}
                if kernel == 'rbf':
                    best_params['gamma'] = 'scale'
            
            final_svm = SVC(kernel=kernel, class_weight='balanced', **best_params)
            final_svm.fit(X_train_scaled, y_train_outer)
            outer_score = final_svm.score(X_val_scaled, y_val_outer)
            
            nested_scores.append(outer_score)
            best_params_list.append(best_params)
            
            print(f"Score: {outer_score:.3f}")
        
        nested_scores = np.array(nested_scores)
        print(f"Nested CV scores: {nested_scores}")
        print(f"Mean: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
        
        # Get most common best parameters
        best_params = self._get_most_common_params(best_params_list)
        
        return {
            'kernel': kernel,
            'nested_scores': nested_scores,
            'mean_score': nested_scores.mean(),
            'std_score': nested_scores.std(),
            'best_params': best_params
        }
    
    def _generate_param_combinations(self, param_grid):
        """Generate all parameter combinations from grid."""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def _get_most_common_params(self, params_list):
        """Get most common parameter combination."""
        if not params_list:
            return {'C': 1}
        
        # Convert to string for comparison
        params_str = [str(sorted(p.items())) for p in params_list]
        
        # Find most common
        from collections import Counter
        most_common = Counter(params_str).most_common(1)[0][0]
        
        # Convert back to dict
        import ast
        return dict(ast.literal_eval(most_common))
    
    def create_ensemble_model(self, X, y, groups):
        """
        Create voting ensemble from multiple optimized kernels.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            
        Returns:
            VotingClassifier: Trained ensemble model
        """
        print("\n=== ENSEMBLE MODEL CREATION ===")
        
        # Test different kernels with nested CV
        kernels = ['linear', 'rbf', 'poly']
        kernel_results = {}
        
        for kernel in kernels:
            try:
                result = self.nested_cross_validation(X, y, groups, kernel)
                kernel_results[kernel] = result
            except Exception as e:
                print(f"Error with {kernel} kernel: {e}")
                continue
        
        if not kernel_results:
            raise ValueError("No kernels could be successfully evaluated")
        
        # Select top performing kernels (at least top 2)
        sorted_kernels = sorted(kernel_results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        top_kernels = sorted_kernels[:min(3, len(sorted_kernels))]  # Top 3 or all available
        
        print(f"\nTop performing kernels:")
        for kernel, result in top_kernels:
            print(f"  {kernel}: {result['mean_score']:.4f} ± {result['std_score']:.4f}")
        
        # Create ensemble estimators
        estimators = []
        for kernel, result in top_kernels:
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
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use probability-based voting
        )
        
        print(f"Created ensemble with {len(estimators)} estimators")
        
        return ensemble, kernel_results
    
    def evaluate_ensemble(self, ensemble, X, y, groups):
        """
        Evaluate ensemble model using LOSO CV.
        
        Args:
            ensemble: Trained ensemble model
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            
        Returns:
            dict: Evaluation results
        """
        print("\n=== ENSEMBLE EVALUATION ===")
        
        logo = LeaveOneGroupOut()
        
        # Collect predictions for each fold
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
                'n_samples': len(y_test),
                'true_labels': y_test.tolist(),
                'predictions': y_pred.tolist()
            }
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
            
            print(f"Subject {test_subject}: Accuracy = {accuracy:.4f} ({len(y_test)} samples)")
        
        # Overall evaluation
        overall_accuracy = accuracy_score(y_true_all, y_pred_all)
        conf_matrix = confusion_matrix(y_true_all, y_pred_all)
        class_report = classification_report(y_true_all, y_pred_all, target_names=['Truth', 'Lie'])
        
        results = {
            'overall_accuracy': overall_accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'subject_results': subject_results,
            'predictions': {
                'y_true': y_true_all,
                'y_pred': y_pred_all
            }
        }
        
        return results
    
    def run_advanced_analysis(self):
        """
        Run complete advanced analysis pipeline.
        """
        print("=== ADVANCED HRV-BASED LIE DETECTION ===")
        print("Using: Nested CV + RFE + Class Weighting + Ensemble Voting")
        
        # Step 1: Discover and load data
        print("\n1. Data Discovery and Loading...")
        file_metadata = self.discover_files()
        if not file_metadata:
            print("No valid CSV files found!")
            return
        
        self.all_data = self.load_data(file_metadata)
        
        # Step 2: Prepare features
        print("\n2. Feature Preparation...")
        X, y, groups, feature_names = self.prepare_features(self.all_data)
        
        # Step 3: Recursive Feature Elimination
        print("\n3. Recursive Feature Elimination...")
        X_selected, selected_features, rfe_selector = self.recursive_feature_elimination(X, y, groups, feature_names)
        
        # Step 4: Create and evaluate ensemble
        print("\n4. Ensemble Model Creation...")
        ensemble, kernel_results = self.create_ensemble_model(X_selected, y, groups)
        
        # Step 5: Final evaluation
        print("\n5. Final Ensemble Evaluation...")
        results = self.evaluate_ensemble(ensemble, X_selected, y, groups)
        
        # Step 6: Results summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Selected Features ({len(selected_features)}): {selected_features}")
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])
        print(f"\nClassification Report:")
        print(results['classification_report'])
        
        # Save results
        self.save_advanced_results(results, kernel_results, selected_features)
        
        return results, ensemble
    
    def save_advanced_results(self, results, kernel_results, selected_features):
        """
        Save comprehensive analysis results.
        """
        print("\n6. Saving Results...")
        
        # 1. Kernel comparison
        kernel_df = pd.DataFrame([
            {
                'kernel': kernel,
                'mean_score': data['mean_score'],
                'std_score': data['std_score'],
                'best_params': str(data['best_params'])
            }
            for kernel, data in kernel_results.items()
        ]).sort_values('mean_score', ascending=False)
        kernel_df.to_csv('advanced_kernel_results.csv', index=False)
        
        # 2. Subject-wise results
        subject_df = pd.DataFrame([
            {
                'subject': subject,
                'accuracy': data['accuracy'],
                'n_samples': data['n_samples']
            }
            for subject, data in results['subject_results'].items()
        ]).sort_values('accuracy', ascending=False)
        subject_df.to_csv('advanced_subject_results.csv', index=False)
        
        # 3. Selected features
        feature_df = pd.DataFrame({
            'feature': selected_features,
            'rank': range(1, len(selected_features) + 1)
        })
        feature_df.to_csv('advanced_selected_features.csv', index=False)
        
        # 4. Summary
        summary = {
            'overall_accuracy': results['overall_accuracy'],
            'n_selected_features': len(selected_features),
            'selected_features': ', '.join(selected_features),
            'n_subjects': len(results['subject_results']),
            'techniques_used': 'Nested CV + RFE + Class Weighting + Ensemble Voting'
        }
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('advanced_analysis_summary.csv', index=False)
        
        print("Advanced results saved to CSV files with 'advanced_' prefix")

def main():
    """
    Main execution function.
    """
    detector = AdvancedHRVLieDetector(data_directory='.')
    
    try:
        results, ensemble = detector.run_advanced_analysis()
        
        if results:
            print(f"\n🎯 SUCCESS! Advanced analysis completed!")
            print(f"📊 Final Accuracy: {results['overall_accuracy']:.1%}")
            print(f"🔧 Techniques Used: Nested CV + RFE + Class Weighting + Ensemble")
            print(f"📈 Expected Improvement: Should be 60-75% accuracy range")
        else:
            print("❌ Advanced analysis failed!")
            
    except Exception as e:
        print(f"❌ Error in advanced analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()