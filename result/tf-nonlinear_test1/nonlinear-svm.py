#!/usr/bin/env python3
"""
HRV-Based Lie Detection using SVM with Combined Original + Validated Features
Combines basic HRV features with research-validated features for comprehensive analysis.
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

class CombinedHRVLieDetector:
    def __init__(self, data_directory='.'):
        """
        Initialize the Combined HRV Lie Detection system.
        
        Args:
            data_directory (str): Directory containing both original and validated CSV files
        """
        self.data_directory = data_directory
        
        # Original HRV features from time and frequency domains
        self.original_time_features = [
            'nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
            'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index'
        ]
        self.original_frequency_features = [
            'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
            'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak'
        ]
        
        # 14 Research-validated HRV features
        self.validated_features = [
            # Time-Frequency Domain (6 features)
            'stft_lf_power', 'stft_hf_power', 'stft_lf_hf_ratio',
            'wt_lf_power', 'wt_hf_power', 'wt_lf_hf_ratio',
            # Non-linear Domain - Poincaré Plot (3 features)
            'sd1', 'sd2', 'sd1_sd2_ratio',
            # Entropy Features (3 features)  
            'apen', 'sampen', 'mse',
            # Fractal Dimension Features (2 features)
            'dfa', 'cd'
        ]
        
        self.all_data = None
        self.best_params = {}
        self.best_scores = {}
        
    def parse_original_filename(self, filename):
        """Parse original CSV filename to extract metadata."""
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
                'filename': filename,
                'file_type': 'original'
            }
        return None
    
    def parse_validated_filename(self, filename):
        """Parse validated CSV filename to extract metadata."""
        # Pattern: pr_{subject}_{condition}_{sequence_type}_validated_windowed.csv
        pattern = r'pr_([^_]+)_([^_]+)_(truth|lie)-sequence_validated_windowed\.csv'
        match = re.match(pattern, filename)
        
        if match:
            subject = match.group(1)
            condition = match.group(2)
            label = match.group(3)  # 'truth' or 'lie'
            
            return {
                'subject': subject,
                'condition': condition,
                'label': label,
                'filename': filename,
                'file_type': 'validated'
            }
        return None
    
    def discover_files(self):
        """
        Discover and organize all CSV files in the directory.
        
        Returns:
            dict: Organized file metadata by type
        """
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        
        original_files = []
        validated_files = []
        
        for filename in csv_files:
            # Try parsing as original file
            original_meta = self.parse_original_filename(filename)
            if original_meta:
                original_files.append(original_meta)
                continue
            
            # Try parsing as validated file
            validated_meta = self.parse_validated_filename(filename)
            if validated_meta:
                validated_files.append(validated_meta)
        
        print(f"Discovered {len(original_files)} original CSV files")
        print(f"Discovered {len(validated_files)} validated CSV files")
        
        # Check subjects and conditions
        original_subjects = set(meta['subject'] for meta in original_files)
        validated_subjects = set(meta['subject'] for meta in validated_files)
        
        common_subjects = original_subjects & validated_subjects
        
        print(f"Original subjects: {sorted(original_subjects)}")
        print(f"Validated subjects: {sorted(validated_subjects)}")
        print(f"Common subjects: {sorted(common_subjects)}")
        
        if not common_subjects:
            raise ValueError("No common subjects found between original and validated files!")
        
        return {
            'original': original_files,
            'validated': validated_files,
            'common_subjects': common_subjects
        }
    
    def load_original_data(self, file_metadata):
        """Load and combine original CSV files."""
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
            return pd.DataFrame()
            
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} original HRV samples")
        
        return combined_df
    
    def load_validated_data(self, file_metadata):
        """Load and combine validated CSV files."""
        all_dataframes = []
        
        for meta in file_metadata:
            filepath = os.path.join(self.data_directory, meta['filename'])
            
            try:
                df = pd.read_csv(filepath)
                
                # Add metadata columns
                df['subject'] = meta['subject']
                df['condition'] = meta['condition']
                df['label'] = meta['label']
                df['binary_label'] = 1 if meta['label'] == 'lie' else 0
                
                all_dataframes.append(df)
                
            except Exception as e:
                print(f"Error loading {meta['filename']}: {e}")
                continue
        
        if not all_dataframes:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Loaded {len(combined_df)} validated HRV samples")
        
        return combined_df
    
    def merge_original_features(self, original_df):
        """Merge time and frequency domain features from original data."""
        if original_df.empty:
            return pd.DataFrame()
        
        # Separate time and frequency data
        time_df = original_df[original_df['domain'] == 'time'].copy()
        freq_df = original_df[original_df['domain'] == 'frequency'].copy()
        
        if time_df.empty or freq_df.empty:
            print("Warning: Missing time or frequency domain data in original files")
            return pd.DataFrame()
        
        # Select columns for merging
        time_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.original_time_features
        freq_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number'] + self.original_frequency_features
        
        # Only keep columns that exist
        time_cols = [col for col in time_cols if col in time_df.columns]
        freq_cols = [col for col in freq_cols if col in freq_df.columns]
        
        time_df_clean = time_df[time_cols].copy()
        freq_df_clean = freq_df[freq_cols].copy()
        
        # Merge on common identifiers
        merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
        merged_df = pd.merge(
            time_df_clean, freq_df_clean, 
            on=merge_cols, 
            suffixes=('_time', '_freq')
        )
        
        print(f"Merged original features: {len(merged_df)} samples")
        return merged_df
    
    def combine_all_features(self, original_merged, validated_df):
        """Combine original and validated features into single dataset."""
        if original_merged.empty or validated_df.empty:
            print("Warning: Cannot combine - missing original or validated data")
            return pd.DataFrame()
        
        # Merge on common identifiers
        merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
        
        # Only keep subjects that exist in both datasets
        original_subjects = set(original_merged['subject'].unique())
        validated_subjects = set(validated_df['subject'].unique())
        common_subjects = original_subjects & validated_subjects
        
        if not common_subjects:
            print("Error: No common subjects between original and validated data")
            return pd.DataFrame()
        
        # Filter to common subjects
        original_filtered = original_merged[original_merged['subject'].isin(common_subjects)].copy()
        validated_filtered = validated_df[validated_df['subject'].isin(common_subjects)].copy()
        
        # Merge datasets
        combined_df = pd.merge(
            original_filtered, validated_filtered,
            on=merge_cols,
            suffixes=('_orig', '_val')
        )
        
        print(f"Combined dataset: {len(combined_df)} samples with both original and validated features")
        print(f"Common subjects: {sorted(common_subjects)}")
        
        return combined_df
    
    def prepare_combined_features(self, df):
        """
        Prepare feature matrix with both original and validated features.
        Uses ALL samples - no filtering.
        
        Args:
            df (pd.DataFrame): Combined dataset
            
        Returns:
            tuple: (X, y, groups, feature_names)
        """
        if df.empty:
            raise ValueError("Empty combined dataset")
        
        # Collect all available features
        all_features = []
        
        # Add original time domain features
        for feature in self.original_time_features:
            if feature in df.columns:
                all_features.append(feature)
            elif f"{feature}_time" in df.columns:
                all_features.append(f"{feature}_time")
        
        # Add original frequency domain features  
        for feature in self.original_frequency_features:
            if feature in df.columns:
                all_features.append(feature)
            elif f"{feature}_freq" in df.columns:
                all_features.append(f"{feature}_freq")
        
        # Add validated features
        for feature in self.validated_features:
            if feature in df.columns:
                all_features.append(feature)
            elif f"{feature}_val" in df.columns:
                all_features.append(f"{feature}_val")
        
        # Check for missing features
        available_features = [col for col in all_features if col in df.columns]
        missing_features = [col for col in all_features if col not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features {missing_features}")
        
        if not available_features:
            raise ValueError("No valid features available for analysis")
        
        print(f"Using ALL {len(df)} samples - no filtering applied")
        
        # Handle NaN values through imputation only (no sample removal)
        df_clean = df.copy()
        
        # Check for NaN values in features
        nan_counts = df_clean[available_features].isnull().sum()
        features_with_nans = nan_counts[nan_counts > 0]
        
        if len(features_with_nans) > 0:
            print(f"Handling NaN values in {len(features_with_nans)} features through imputation:")
            
            from sklearn.impute import SimpleImputer
            
            # Use median imputation for all NaN values
            imputer = SimpleImputer(strategy='median')
            
            # Apply imputation to all features
            feature_data = df_clean[available_features].copy()
            feature_data_imputed = imputer.fit_transform(feature_data)
            
            # Update the dataframe with imputed values
            df_clean[available_features] = feature_data_imputed
            
            print(f"  Successfully imputed NaN values using median strategy")
        else:
            print("No NaN values found in feature data")
        
        # Use all samples
        X = df_clean[available_features].values
        y = df_clean['binary_label'].values  
        groups = df_clean['subject'].values
        
        print(f"\nFinal dataset statistics:")
        print(f"  Total samples: {len(df_clean)} (using ALL samples)")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Total features: {len(available_features)}")
        print(f"    - Original time features: {len([f for f in available_features if any(orig in f for orig in self.original_time_features)])}")
        print(f"    - Original frequency features: {len([f for f in available_features if any(orig in f for orig in self.original_frequency_features)])}")
        print(f"    - Validated features: {len([f for f in available_features if f in self.validated_features or f.replace('_val', '') in self.validated_features])}")
        print(f"  Label distribution: Truth={np.sum(y==0)}, Lie={np.sum(y==1)}")
        print(f"  Balance ratio: {max(np.sum(y==0), np.sum(y==1)) / (min(np.sum(y==0), np.sum(y==1)) + 1e-8):.2f}")
        print(f"  Subjects: {sorted(set(groups))}")
        
        return X, y, groups, available_features
    
    def grid_search_svm(self, X, y, groups):
        """
        Perform FAST grid search for SVM hyperparameters with LOSOCV.
        Optimized for speed while maintaining performance.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            groups (np.array): Subject groups
            
        Returns:
            dict: Best parameters and scores for each kernel
        """
        # Define SMALLER parameter grids for faster execution
        param_grids = {
            'linear': {
                'svm__C': [0.1, 1, 10]  # Reduced from 7 to 3 values
            },
            'rbf': {
                'svm__C': [0.1, 1, 10],  # Reduced from 7 to 3 values
                'svm__gamma': ['scale', 0.01, 0.1]  # Reduced from 6 to 3 values
            },
            'poly': {
                'svm__C': [0.1, 1, 10],  # Reduced from 6 to 3 values
                'svm__gamma': ['scale', 0.01],  # Reduced from 5 to 2 values
                'svm__degree': [2, 3]  # Reduced from 3 to 2 values
            }
            # Skip sigmoid kernel for speed - it rarely performs best
        }
        
        # Leave-One-Subject-Out cross-validation
        logo = LeaveOneGroupOut()
        
        kernel_results = {}
        
        for kernel, param_grid in param_grids.items():
            print(f"\nTesting {kernel} kernel...")
            
            # Calculate total combinations for progress tracking
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            total_fits = total_combinations * logo.get_n_splits(X, y, groups)
            print(f"  Will test {total_combinations} combinations × {logo.get_n_splits(X, y, groups)} folds = {total_fits} fits")
            
            # Create pipeline with scaling and SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(kernel=kernel, random_state=42, class_weight='balanced'))
            ])
            
            # Grid search with LOSOCV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=logo,
                scoring='accuracy',
                n_jobs=-1,  # Use all CPU cores
                verbose=1
            )
            
            try:
                import time
                start_time = time.time()
                
                grid_search.fit(X, y, groups=groups)
                
                end_time = time.time()
                duration = end_time - start_time
                
                kernel_results[kernel] = {
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_,
                    'duration': duration
                }
                
                print(f"{kernel} - Best score: {grid_search.best_score_:.4f}")
                print(f"{kernel} - Best params: {grid_search.best_params_}")
                print(f"{kernel} - Duration: {duration:.1f} seconds")
                
            except Exception as e:
                print(f"Error with {kernel} kernel: {e}")
                continue
        
        return kernel_results
    
    def evaluate_model(self, X, y, groups, best_kernel, best_params):
        """
        Evaluate the best model using LOSOCV.
        """
        logo = LeaveOneGroupOut()
        
        # Create pipeline with best parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(**best_params, kernel=best_kernel, random_state=42, class_weight='balanced'))
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
    
    def run_combined_analysis(self):
        """
        Run complete analysis with combined original + validated features.
        """
        print("=== COMBINED HRV LIE DETECTION ANALYSIS ===")
        print("Combining Original HRV + Validated Research Features")
        
        # Step 1: Discover files
        print("\n1. Discovering CSV files...")
        file_info = self.discover_files()
        
        # Step 2: Load original data
        print("\n2. Loading original HRV data...")
        original_df = self.load_original_data(file_info['original'])
        
        # Step 3: Load validated data
        print("\n3. Loading validated HRV data...")
        validated_df = self.load_validated_data(file_info['validated'])
        
        # Step 4: Merge original features
        print("\n4. Merging original time + frequency features...")
        original_merged = self.merge_original_features(original_df)
        
        # Step 5: Combine all features
        print("\n5. Combining original + validated features...")
        self.all_data = self.combine_all_features(original_merged, validated_df)
        
        if self.all_data.empty:
            print("Error: Could not create combined dataset!")
            return
        
        # Step 6: Prepare combined features
        print("\n6. Preparing combined feature matrix...")
        X, y, groups, feature_names = self.prepare_combined_features(self.all_data)
        
        # Step 7: Grid search for best kernel and parameters
        print("\n7. Performing grid search with LOSOCV...")
        kernel_results = self.grid_search_svm(X, y, groups)
        
        if not kernel_results:
            print("No successful kernel evaluations!")
            return
        
        # Step 8: Select best kernel
        best_kernel = max(kernel_results.keys(), key=lambda k: kernel_results[k]['best_score'])
        best_params = kernel_results[best_kernel]['best_params']
        
        print(f"\n8. Best kernel: {best_kernel}")
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {kernel_results[best_kernel]['best_score']:.4f}")
        
        # Step 9: Final evaluation
        print("\n9. Final model evaluation...")
        
        # Extract SVM parameters from pipeline parameters
        svm_params = {k.replace('svm__', ''): v for k, v in best_params.items()}
        # Don't add class_weight here since it's already in the evaluate_model method
        
        results = self.evaluate_model(X, y, groups, best_kernel, svm_params)
        
        # Step 10: Display results
        print("\n" + "="*80)
        print("COMBINED FEATURES FINAL RESULTS")
        print("="*80)
        print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        print(f"Total Features Used: {len(feature_names)}")
        print(f"Best Kernel: {best_kernel}")
        print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
        print(f"\nClassification Report:\n{results['classification_report']}")
        
        # Save results
        self.save_combined_results(results, kernel_results, feature_names)
        
        return results
    
    def save_combined_results(self, results, kernel_results, feature_names):
        """Save combined analysis results to CSV files."""
        print(f"\n10. Saving combined results...")
        
        # 1. Kernel comparison results
        kernel_df = pd.DataFrame([
            {
                'kernel': kernel,
                'best_score': data['best_score'],
                'best_params': str(data['best_params'])
            }
            for kernel, data in kernel_results.items()
        ]).sort_values('best_score', ascending=False)
        
        kernel_df.to_csv('combined_kernel_comparison.csv', index=False)
        
        # 2. Subject-wise results
        subject_df = pd.DataFrame([
            {
                'subject': subject,
                'accuracy': data['accuracy'],
                'n_samples': data['n_samples']
            }
            for subject, data in results['subject_results'].items()
        ]).sort_values('accuracy', ascending=False)
        
        subject_df.to_csv('combined_subject_results.csv', index=False)
        
        # 3. Feature list
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'feature_type': ['original' if any(orig in f for orig in self.original_time_features + self.original_frequency_features) 
                           else 'validated' for f in feature_names],
            'index': range(len(feature_names))
        })
        feature_df.to_csv('combined_features_used.csv', index=False)
        
        # 4. Summary results
        summary = {
            'analysis_type': 'combined_original_validated',
            'best_kernel': results['best_kernel'],
            'best_params': str(results['best_params']),
            'overall_accuracy': results['overall_accuracy'],
            'total_features': len(feature_names),
            'original_features': len([f for f in feature_names if any(orig in f for orig in self.original_time_features + self.original_frequency_features)]),
            'validated_features': len([f for f in feature_names if f in self.validated_features or f.replace('_val', '') in self.validated_features]),
            'n_subjects': len(results['subject_results'])
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv('combined_analysis_summary.csv', index=False)
        
        print("Combined results saved:")
        print("  - combined_kernel_comparison.csv")
        print("  - combined_subject_results.csv") 
        print("  - combined_features_used.csv")
        print("  - combined_analysis_summary.csv")

def main():
    """
    Main execution function.
    """
    detector = CombinedHRVLieDetector(data_directory='.')
    
    try:
        results = detector.run_combined_analysis()
        
        if results:
            accuracy = results['overall_accuracy']
            n_subjects = len(results['subject_results'])
            
            print(f"\n🎯 COMBINED FEATURE ANALYSIS COMPLETE!")
            print(f"📊 Final Accuracy: {accuracy:.1%}")
            print(f"👥 Subjects Analyzed: {n_subjects}")
            
            if accuracy > 0.7:
                print(f"🚀 EXCELLENT! Combined features achieved >70% accuracy!")
            elif accuracy > 0.6:
                print(f"✅ GOOD! Combined features achieved >60% accuracy!")
            elif accuracy > 0.55:
                print(f"📈 IMPROVED! Better performance with combined features!")
            else:
                print(f"📋 Results available for further analysis")
        else:
            print("❌ Analysis failed!")
            
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()