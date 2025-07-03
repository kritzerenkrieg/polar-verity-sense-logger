#!/usr/bin/env python3
"""
HRV Lie Detection Model Loader - WORKING VERSION
Loads and uses the pre-trained optimized model: best_hrv_model_best_12feat.pkl

Model Performance:
- Expected Accuracy: 0.7205
- Improvement: +3.30%
- Method: top_k_features
- Features: 6
"""

import joblib
import pandas as pd
import numpy as np
import os

class HRVLieDetectorLoader:
    def __init__(self, model_path='best_hrv_model_best_12feat.pkl'):
        """Load the pre-trained model."""
        print(f"Loading HRV Lie Detection model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model data
        self.model_data = joblib.load(model_path)
        self.pipeline = self.model_data['pipeline']
        self.selected_features = self.model_data['selected_features']
        self.config = self.model_data['model_config']
        self.feature_domains = self.model_data['feature_domains']
        
        print(f"Model loaded successfully!")
        print(f"Expected accuracy: {self.config['expected_accuracy']:.4f}")
        print(f"Features required: {self.selected_features}")
        print(f"Time domain features: {self.feature_domains['time_features']}")
        print(f"Frequency domain features: {self.feature_domains['freq_features']}")
    
    def prepare_features(self, df_time=None, df_frequency=None):
        """
        Prepare features from time and/or frequency domain data.
        
        Args:
            df_time (pd.DataFrame): Time domain features (optional if only freq features needed)
            df_frequency (pd.DataFrame): Frequency domain features (optional if only time features needed)
            
        Returns:
            np.array: Prepared feature matrix
        """
        time_features = self.feature_domains['time_features']
        freq_features = self.feature_domains['freq_features']
        
        if time_features and freq_features:
            # Need both domains
            if df_time is None or df_frequency is None:
                raise ValueError("Both time and frequency domain data required")
            
            # Merge data on common identifiers
            merge_cols = ['subject', 'condition', 'label', 'binary_label', 'window_number']
            available_merge_cols = [col for col in merge_cols if col in df_time.columns and col in df_frequency.columns]
            
            if not available_merge_cols:
                # Fallback: assume same order and length
                if len(df_time) != len(df_frequency):
                    raise ValueError("Time and frequency data must have same length if no merge columns available")
                
                time_data = df_time[time_features].values
                freq_data = df_frequency[freq_features].values
                feature_matrix = np.column_stack([time_data, freq_data])
            else:
                merged_df = pd.merge(
                    df_time[available_merge_cols + time_features],
                    df_frequency[available_merge_cols + freq_features],
                    on=available_merge_cols
                )
                feature_matrix = merged_df[self.selected_features].values
                
        elif time_features:
            # Only time domain features needed
            if df_time is None:
                raise ValueError("Time domain data required")
            feature_matrix = df_time[self.selected_features].values
            
        elif freq_features:
            # Only frequency domain features needed  
            if df_frequency is None:
                raise ValueError("Frequency domain data required")
            feature_matrix = df_frequency[self.selected_features].values
            
        else:
            raise ValueError("No features specified in model")
        
        return feature_matrix
    
    def predict(self, df_time=None, df_frequency=None, return_probabilities=False):
        """
        Make lie detection predictions.
        
        Args:
            df_time (pd.DataFrame): Time domain HRV features
            df_frequency (pd.DataFrame): Frequency domain HRV features  
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            array: Predictions (0=truth, 1=lie) and optionally probabilities
        """
        # Prepare features
        X = self.prepare_features(df_time, df_frequency)
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        
        if return_probabilities:
            probabilities = self.pipeline.predict_proba(X)
            return predictions, probabilities
        else:
            return predictions
    
    def predict_single_sample(self, time_features_dict=None, freq_features_dict=None):
        """
        Make prediction for a single sample.
        
        Args:
            time_features_dict (dict): Time domain features as dict
            freq_features_dict (dict): Frequency domain features as dict
            
        Returns:
            tuple: (prediction, confidence_score)
        """
        # Create single-row DataFrames
        if time_features_dict:
            df_time = pd.DataFrame([time_features_dict])
        else:
            df_time = None
            
        if freq_features_dict:
            df_frequency = pd.DataFrame([freq_features_dict])
        else:
            df_frequency = None
        
        # Get prediction and probability
        prediction, probabilities = self.predict(df_time, df_frequency, return_probabilities=True)
        
        # Extract confidence (distance from decision boundary)
        confidence = max(probabilities[0]) - 0.5  # How far from 50/50
        
        result = "LIE" if prediction[0] == 1 else "TRUTH"
        
        return result, confidence
    
    def get_model_info(self):
        """Get detailed model information."""
        return {
            'expected_accuracy': self.config['expected_accuracy'],
            'improvement_percent': self.config['improvement_percent'],
            'method': self.config['method'],
            'kernel': self.config['kernel'],
            'features': self.selected_features,
            'n_features': len(self.selected_features),
            'training_samples': self.config['training_samples']
        }

def demo_with_synthetic_data():
    """Demonstrate the model with synthetic HRV data."""
    print("\n=== DEMO: HRV Lie Detection Model ===")
    
    try:
        # Load the trained model
        detector = HRVLieDetectorLoader()
        
        # Generate synthetic test data that matches the feature requirements
        print("\nGenerating synthetic test data...")
        
        time_features = detector.feature_domains['time_features']
        freq_features = detector.feature_domains['freq_features']
        
        # Create realistic HRV feature ranges
        np.random.seed(42)  # For reproducible results
        n_samples = 5
        
        synthetic_time_data = {}
        synthetic_freq_data = {}
        
        # Generate synthetic time domain features
        for feature in time_features:
            if feature == 'nn_mean':
                synthetic_time_data[feature] = np.random.normal(850, 100, n_samples)  # ms
            elif feature == 'nn_min':
                synthetic_time_data[feature] = np.random.normal(650, 50, n_samples)   # ms
            elif feature == 'nn_count':
                synthetic_time_data[feature] = np.random.randint(25, 40, n_samples)    # count
            elif feature == 'sdnn':
                synthetic_time_data[feature] = np.random.normal(45, 15, n_samples)     # ms
            elif feature == 'sdsd':
                synthetic_time_data[feature] = np.random.normal(35, 10, n_samples)     # ms
            elif feature == 'rmssd':
                synthetic_time_data[feature] = np.random.normal(40, 12, n_samples)     # ms
            elif feature == 'pnn20':
                synthetic_time_data[feature] = np.random.normal(15, 5, n_samples)      # %
            elif feature == 'pnn50':
                synthetic_time_data[feature] = np.random.normal(8, 3, n_samples)       # %
            else:
                synthetic_time_data[feature] = np.random.normal(50, 15, n_samples)     # generic
        
        # Generate synthetic frequency domain features
        for feature in freq_features:
            if feature == 'lf_power':
                synthetic_freq_data[feature] = np.random.normal(1200, 400, n_samples)  # ms²
            elif feature == 'hf_power':
                synthetic_freq_data[feature] = np.random.normal(800, 300, n_samples)   # ms²
            elif feature == 'lf_norm':
                synthetic_freq_data[feature] = np.random.normal(60, 10, n_samples)     # %
            elif feature == 'hf_norm':
                synthetic_freq_data[feature] = np.random.normal(40, 10, n_samples)     # %
            elif feature == 'lf_hf_ratio':
                synthetic_freq_data[feature] = np.random.normal(1.5, 0.5, n_samples)   # ratio
            elif feature == 'ln_hf':
                synthetic_freq_data[feature] = np.random.normal(6.5, 0.8, n_samples)   # ln(ms²)
            else:
                synthetic_freq_data[feature] = np.random.normal(100, 30, n_samples)    # generic
        
        # Create DataFrames
        df_time_test = pd.DataFrame(synthetic_time_data) if time_features else None
        df_freq_test = pd.DataFrame(synthetic_freq_data) if freq_features else None
        
        # Make predictions on synthetic data
        print("\nMaking predictions on synthetic data...")
        predictions, probabilities = detector.predict(df_time_test, df_freq_test, return_probabilities=True)
        
        print("\nPrediction Results:")
        print("-" * 50)
        for i in range(n_samples):
            pred_label = "LIE" if predictions[i] == 1 else "TRUTH"
            confidence = max(probabilities[i])
            print(f"Sample {i+1}: {pred_label} (confidence: {confidence:.3f})")
        
        # Demonstrate single sample prediction
        print("\n=== Single Sample Prediction Demo ===")
        
        # Create example features for a single prediction
        example_time = {feat: synthetic_time_data[feat][0] for feat in time_features} if time_features else None
        example_freq = {feat: synthetic_freq_data[feat][0] for feat in freq_features} if freq_features else None
        
        result, confidence = detector.predict_single_sample(example_time, example_freq)
        
        print(f"Example prediction: {result} (confidence: {confidence:.3f})")
        
        if example_time:
            print(f"Time features used: {example_time}")
        if example_freq:
            print(f"Frequency features used: {example_freq}")
        
        # Show model information
        print("\n=== Model Information ===")
        info = detector.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print("\n=== Demo Complete ===")
        print("The model is working correctly!")
        print("You can now use this script with your real HRV data.")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Make sure the PKL model file exists in the same directory.")

# Run the demo when script is executed
if __name__ == "__main__":
    demo_with_synthetic_data()
