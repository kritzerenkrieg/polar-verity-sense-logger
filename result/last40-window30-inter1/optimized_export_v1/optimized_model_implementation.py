#!/usr/bin/env python3
"""
HRV Lie Detection - Optimized Implementation
Best result: 0.7205 accuracy
Improvement: +3.30%
Method: top_k_features
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

class OptimizedHRVLieDetector:
    def __init__(self):
        self.selected_features = ['nn_count', 'pnn20', 'hf_power', 'lf_hf_ratio', 'lf_norm', 'hf_norm']
        self.kernel = 'rbf'
        self.kernel_params = {'svm__C': 10, 'svm__gamma': 0.1}
        self.expected_accuracy = 0.7205
        
        # Initialize pipeline
        svm_params = {k.replace('svm__', ''): v for k, v in self.kernel_params.items()}
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=self.kernel, **svm_params, random_state=42, probability=True))
        ])
        
        self.is_trained = False
    
    def prepare_features(self, df_time, df_frequency):
        """Prepare features from time and frequency domain data."""
        time_features = [f for f in self.selected_features 
                        if f in ['nn_count', 'nn_mean', 'nn_min', 'nn_max', 'sdnn', 
                               'sdsd', 'rmssd', 'pnn20', 'pnn50', 'triangular_index']]
        freq_features = [f for f in self.selected_features
                        if f in ['lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 
                               'hf_norm', 'ln_hf', 'lf_peak', 'hf_peak']]
        
        if time_features and freq_features:
            merged_df = pd.merge(
                df_time[['subject', 'condition', 'label', 'binary_label', 'window_number'] + time_features],
                df_frequency[['subject', 'condition', 'label', 'binary_label', 'window_number'] + freq_features],
                on=['subject', 'condition', 'label', 'binary_label', 'window_number']
            )
            return merged_df[self.selected_features].values
        elif time_features:
            return df_time[self.selected_features].values
        else:
            return df_frequency[self.selected_features].values
    
    def train(self, df_time, df_frequency, labels):
        """Train the optimized model."""
        X = self.prepare_features(df_time, df_frequency)
        self.pipeline.fit(X, labels)
        self.is_trained = True
        print(f"Model trained! Expected accuracy: {self.expected_accuracy:.4f}")
    
    def predict(self, df_time, df_frequency, return_probabilities=False):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X = self.prepare_features(df_time, df_frequency)
        predictions = self.pipeline.predict(X)
        
        if return_probabilities:
            probabilities = self.pipeline.predict_proba(X)
            return predictions, probabilities
        return predictions
    
    def save_model(self, filename='optimized_hrv_model.pkl'):
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        joblib.dump({
            'pipeline': self.pipeline,
            'selected_features': self.selected_features,
            'config': {
                'kernel': self.kernel,
                'params': self.kernel_params,
                'accuracy': self.expected_accuracy
            }
        }, filename)
        print(f"Model saved to {filename}")

# Usage example:
# detector = OptimizedHRVLieDetector()
# detector.train(df_time, df_frequency, labels)
# predictions = detector.predict(new_df_time, new_df_frequency)
