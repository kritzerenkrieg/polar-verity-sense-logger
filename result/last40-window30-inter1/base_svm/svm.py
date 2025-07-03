#!/usr/bin/env python3
"""
HRV-Based Lie Detection using SVM with Leave-One-Subject-Out Cross-Validation
Automatically processes CSV files and performs kernel selection with grid search.
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
import warnings
warnings.filterwarnings('ignore')

class HRVLieDetector:
    def __init__(self, data_directory='.'):
        """
        Initialize the HRV Lie Detection system.
        
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
        
        # Original complete feature sets (kept for reference)
        self.time_domain_features = [
            'nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
            'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index'
        ]
        self.frequency_domain_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
            'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak'
        ]
        
        self.all_data = None
        self.best_params = {}
        self.best_scores = {}
        
    def parse_filename(self, filename):
        """
        Parse CSV filename to extract metadata.
        
        Args:
            filename (str): CSV filename
            
        Returns:
            dict: Parsed metadata or None if invalid format
        """
        # Pattern: pr_{subject}_{condition}_{sequence_type}_windowed.csv
        # or pr_{subject}_{condition}_{sequence_type}_frequency_windowed.csv
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
        
        # Group by subjects and conditions
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
    
    def prepare_features(self, df, domain='both', use_selected_only=True):
        """
        Prepare feature matrix based on domain selection.
        
        Args:
            df (pd.DataFrame): Combined dataset
            domain (str): 'time', 'frequency', or 'both'
            use_selected_only (bool): Use only high-reliability features
            
        Returns:
            tuple: (X, y, groups, feature_names)
        """
        if use_selected_only:
            # Use only high-reliability features
            if domain == 'time':
                feature_cols = self.selected_features['time']
                df_filtered = df[df['domain'] == 'time'].copy()
            elif domain == 'frequency':
                feature_cols = self.selected_features['frequency']
                df_filtered = df[df['domain'] == 'frequency'].copy()
            elif domain == 'both':
                # Merge time and frequency data for same subject/condition/label
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
                
                # Create feature column names - no suffix needed for time features since they're unique
                # but frequency features will have _freq suffix if there are conflicts
                time_features = self.selected_features['time']  # nn_mean
                freq_features = [f + '_freq' if f in time_df_clean.columns and f in freq_df_clean.columns 
                               else f for f in self.selected_features['frequency']]
                feature_cols = time_features + freq_features
                
                print(f"Merged {len(time_df)} time samples with {len(freq_df)} frequency samples")
                print(f"Result: {len(df_filtered)} combined samples")
        else:
            # Use all features (original behavior)
            if domain == 'time':
                feature_cols = self.time_domain_features
                df_filtered = df[df['domain'] == 'time'].copy()
            elif domain == 'frequency':
                feature_cols = self.frequency_domain_features
                df_filtered = df[df['domain'] == 'frequency'].copy()
            elif domain == 'both':
                # Merge time and frequency data for same subject/condition/label
                time_df = df[df['domain'] == 'time'].copy()
                freq_df = df[df['domain'] == 'frequency'].copy()
                
                # Select only needed columns to avoid conflicts
                time_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.time_domain_features
                freq_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.frequency_domain_features
                
                # Only keep columns that exist
                time_cols = [col for col in time_cols if col in time_df.columns]
                freq_cols = [col for col in freq_cols if col in freq_df.columns]
                
                time_df_clean = time_df[time_cols].copy()
                freq_df_clean = freq_df[freq_cols].copy()
                
                # Merge on common identifiers
                merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
                df_filtered = pd.merge(
                    time_df_clean, freq_df_clean, 
                    on=merge_cols, 
                    suffixes=('_time', '_freq')
                )
                
                # Create feature column names with proper suffixes
                time_features = [f + '_time' if f in freq_df_clean.columns else f for f in self.time_domain_features]
                freq_features = [f + '_freq' if f in time_df_clean.columns else f for f in self.frequency_domain_features]
                feature_cols = time_features + freq_features
                
                print(f"Merged {len(time_df)} time samples with {len(freq_df)} frequency samples")
                print(f"Result: {len(df_filtered)} combined samples")
            else:
                raise ValueError("Domain must be 'time', 'frequency', or 'both'")
        
        # Check for missing features
        available_features = [col for col in feature_cols if col in df_filtered.columns]
        missing_features = [col for col in feature_cols if col not in df_filtered.columns]
        
        if missing_features:
            print(f"Warning: Missing features {missing_features}")
        
        # Prepare final dataset
        df_clean = df_filtered.dropna(subset=available_features)
        
        X = df_clean[available_features].values
        
        # Handle column naming - for merged data, the original columns should be preserved
        # since we merged on them
        y = df_clean['binary_label'].values
        groups = df_clean['subject'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Using {len(available_features)} high-reliability features: {available_features}")
        
        return X, y, groups, available_features
    
    def grid_search_svm(self, X, y, groups):
        """
        Perform grid search for SVM hyperparameters with LOSOCV.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            
        Returns:
            dict: Best parameters and scores for each kernel
        """
        # Define parameter grids for different kernels
        param_grids = {
            'linear': {
                'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            },
            'rbf': {
                'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'poly': {
                'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'svm__degree': [2, 3, 4]
            },
            'sigmoid': {
                'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        }
        
        # Leave-One-Subject-Out cross-validation
        logo = LeaveOneGroupOut()
        
        kernel_results = {}
        
        for kernel, param_grid in param_grids.items():
            print(f"\nTesting {kernel} kernel...")
            
            # Create pipeline with scaling and SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, random_state=42))
            ])
            
            # Grid search with LOSOCV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=logo,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            try:
                grid_search.fit(X, y, groups=groups)
                
                kernel_results[kernel] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                print(f"{kernel} - Best score: {grid_search.best_score_:.4f}")
                print(f"{kernel} - Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"Error with {kernel} kernel: {e}")
                continue
        
        return kernel_results
    
    def evaluate_model(self, X, y, groups, best_kernel, best_params):
        """
        Evaluate the best model using LOSOCV.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            best_kernel (str): Best performing kernel
            best_params (dict): Best hyperparameters
            
        Returns:
            dict: Detailed evaluation results
        """
        logo = LeaveOneGroupOut()
        
        # Create pipeline with best parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**best_params, kernel=best_kernel, random_state=42))
        ])
        
        # Collect predictions for each fold
        y_true_all = []
        y_pred_all = []
        subject_results = {}
        
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = groups[test_idx][0]  # All test samples are from same subject
            
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
            'best_kernel': best_kernel,
            'best_params': best_params
        }
        
        return results
    
    def run_analysis(self, domain='both', use_selected_only=True):
        """
        Run complete analysis pipeline.
        
        Args:
            domain (str): 'time', 'frequency', or 'both'
            use_selected_only (bool): Use only high-reliability features
        """
        feature_type = "selected" if use_selected_only else "all"
        print("=== HRV-Based Lie Detection Analysis ===")
        print(f"Domain: {domain}")
        print(f"Features: {feature_type} ({'6 high-reliability features' if use_selected_only else 'all available features'})")
        
        # Step 1: Discover files
        print("\n1. Discovering CSV files...")
        file_metadata = self.discover_files()
        
        if not file_metadata:
            print("No valid CSV files found!")
            return
        
        # Step 2: Load data
        print("\n2. Loading data...")
        self.all_data = self.load_data(file_metadata)
        
        # Step 3: Prepare features
        print("\n3. Preparing features...")
        X, y, groups, feature_names = self.prepare_features(self.all_data, domain, use_selected_only)
        
        # Step 4: Grid search for best kernel and parameters
        print("\n4. Performing grid search with LOSOCV...")
        kernel_results = self.grid_search_svm(X, y, groups)
        
        if not kernel_results:
            print("No successful kernel evaluations!")
            return
        
        # Step 5: Select best kernel
        best_kernel = max(kernel_results.keys(), key=lambda k: kernel_results[k]['best_score'])
        best_params = kernel_results[best_kernel]['best_params']
        
        print(f"\n5. Best kernel: {best_kernel}")
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {kernel_results[best_kernel]['best_score']:.4f}")
        
        # Step 6: Final evaluation
        print("\n6. Final model evaluation...")
        
        # Extract SVM parameters from pipeline parameters
        svm_params = {k.replace('svm__', ''): v for k, v in best_params.items()}
        
        results = self.evaluate_model(X, y, groups, best_kernel, svm_params)
        
        # Step 7: Display results
        print("\n=== FINAL RESULTS ===")
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
        print(f"\nClassification Report:\n{results['classification_report']}")
        
        # Step 8: Feature analysis
        print(f"\n=== FEATURE ANALYSIS ===")
        if use_selected_only:
            print("High-reliability features used:")
            for i, feature in enumerate(feature_names):
                base_feature = feature.replace('_x', '').replace('_y', '')
                if base_feature in ['lf_power', 'ln_hf', 'lf_norm', 'lf_hf_ratio']:
                    direction = "Truth Higher"
                elif base_feature in ['nn_mean', 'hf_norm']:
                    direction = "Lie Higher"
                else:
                    direction = "Unknown"
                print(f"  {i+1}. {feature} ({direction})")
        
        # Save results
        suffix = f"{domain}_{'selected' if use_selected_only else 'all'}"
        self.save_results(results, kernel_results, feature_names, suffix)
        
        return results
    
    def save_results(self, results, kernel_results, feature_names, suffix):
        """
        Save analysis results to CSV files.
        """
        print(f"\n7. Saving results...")
        
        # 1. Kernel comparison results
        kernel_df = pd.DataFrame([
            {
                'kernel': kernel,
                'best_score': data['best_score'],
                'best_params': str(data['best_params'])
            }
            for kernel, data in kernel_results.items()
        ]).sort_values('best_score', ascending=False)
        
        kernel_df.to_csv(f'kernel_comparison_{suffix}.csv', index=False)
        
        # 2. Subject-wise results
        subject_df = pd.DataFrame([
            {
                'subject': subject,
                'accuracy': data['accuracy'],
                'n_samples': data['n_samples']
            }
            for subject, data in results['subject_results'].items()
        ]).sort_values('accuracy', ascending=False)
        
        subject_df.to_csv(f'subject_results_{suffix}.csv', index=False)
        
        # 3. Feature importance (if available)
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'index': range(len(feature_names))
        })
        feature_df.to_csv(f'features_used_{suffix}.csv', index=False)
        
        # 4. Summary results
        summary = {
            'analysis_type': suffix,
            'best_kernel': results['best_kernel'],
            'best_params': str(results['best_params']),
            'overall_accuracy': results['overall_accuracy'],
            'n_features': len(feature_names),
            'n_subjects': len(results['subject_results']),
            'features_used': ', '.join(feature_names)
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f'analysis_summary_{suffix}.csv', index=False)
        
        print(f"Results saved to CSV files with suffix '_{suffix}.csv'")

def main():
    """
    Main execution function.
    """
    # Initialize detector
    detector = HRVLieDetector(data_directory='.')
    
    # Run analysis with selected high-reliability features only
    print(f"\n{'='*80}")
    print(f"ANALYZING WITH HIGH-RELIABILITY FEATURES (6 features)")
    print(f"{'='*80}")
    
    try:
        results = detector.run_analysis(domain='both', use_selected_only=True)
        
        if results:
            print(f"\nHigh-reliability feature analysis completed successfully!")
            print(f"Features used:")
            print(f"  • lf_power (90% consistency, Truth Higher)")
            print(f"  • ln_hf (90% consistency, Truth Higher)")
            print(f"  • nn_mean (80% consistency, Lie Higher)")
            print(f"  • hf_norm (80% consistency, Lie Higher)")
            print(f"  • lf_norm (80% consistency, Truth Higher)")
            print(f"  • lf_hf_ratio (80% consistency, Truth Higher)")
        else:
            print(f"\nHigh-reliability feature analysis failed!")
            
    except Exception as e:
        print(f"Error in high-reliability feature analysis: {e}")
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: ANALYZING WITH ALL FEATURES")
    print(f"{'='*80}")
    
    # For comparison, also run with all features
    try:
        results_all = detector.run_analysis(domain='both', use_selected_only=False)
        
        if results_all:
            print(f"\nAll features analysis completed successfully!")
        else:
            print(f"\nAll features analysis failed!")
            
    except Exception as e:
        print(f"Error in all features analysis: {e}")

if __name__ == "__main__":
    main()