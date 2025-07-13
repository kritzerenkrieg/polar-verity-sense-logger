#!/usr/bin/env python3
"""
HRV Model Prediction Script
Load the trained PKL model and make predictions on new HRV data using majority vote aggregation.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class HRVPredictor:
    def __init__(self, model_path):
        """
        Load the trained HRV model.
        
        Args:
            model_path (str): Path to the .pkl model file
        """
        self.model_data = joblib.load(model_path)
        self.pipeline = self.model_data['pipeline']
        self.selected_features = self.model_data['selected_features']
        self.model_config = self.model_data['model_config']
        self.feature_domains = self.model_data['feature_domains']
        
        print(f"Model loaded successfully!")
        print(f"Expected features: {self.selected_features}")
        print(f"Model accuracy: {self.model_config['expected_accuracy']:.4f}")
        print(f"Feature count: {len(self.selected_features)}")
    
    def get_required_features(self):
        """Return list of required features for prediction."""
        return self.selected_features.copy()
    
    def validate_input(self, data):
        """
        Validate that input data contains all required features.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            bool: True if valid, False otherwise
        """
        available_features = list(data.columns)
        
        missing_features = [f for f in self.selected_features if f not in available_features]
        if missing_features:
            print(f"Error: Missing required features: {missing_features}")
            return False
        
        return True
    
    def prepare_input(self, data):
        """
        Prepare input data for prediction.
        
        Args:
            data (pd.DataFrame): Input HRV features
            
        Returns:
            np.array: Prepared feature array ready for prediction
        """
        if not self.validate_input(data):
            raise ValueError("Input validation failed")
        
        # Extract features in correct order
        feature_array = data[self.selected_features].values
        
        return feature_array
    
    def load_csv_data(self, csv_path):
        """
        Load CSV data in the specific format with all columns.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated data
        """
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            
            # Show available columns vs required features
            available_features = [f for f in self.selected_features if f in df.columns]
            missing_features = [f for f in self.selected_features if f not in df.columns]
            
            print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            print(f"Available required features: {available_features}")
            if missing_features:
                print(f"Missing required features: {missing_features}")
                raise ValueError(f"CSV missing required features: {missing_features}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Failed to load CSV: {e}")
    
    def predict_from_csv(self, csv_path):
        """
        Load CSV and make majority vote prediction.
        
        Args:
            csv_path (str): Path to CSV file
        
        Returns:
            dict: Prediction results using majority vote
        """
        # Load CSV data
        df = self.load_csv_data(csv_path)
        
        # Get majority vote prediction
        result = self.predict_majority_vote(df)
        result['csv_info'] = {
            'file_path': csv_path,
            'total_windows': len(df),
            'aggregation_method': 'majority_vote'
        }
        return result
    
    def predict(self, data, return_probability=True):
        """
        Make prediction on HRV data.
        
        Args:
            data: Input HRV features (DataFrame)
            return_probability (bool): If True, return prediction probabilities
            
        Returns:
            dict: Prediction results
        """
        try:
            # Prepare input
            X = self.prepare_input(data)
            
            # Make predictions
            predictions = self.pipeline.predict(X)
            
            results = {
                'predictions': predictions,
                'labels': ['truth' if p == 0 else 'lie' for p in predictions],
                'input_shape': X.shape,
                'n_samples': len(predictions)
            }
            
            # Add probabilities if available
            if return_probability and hasattr(self.pipeline.named_steps['svm'], 'predict_proba'):
                probabilities = self.pipeline.predict_proba(X)
                results['probabilities'] = probabilities
                results['confidence'] = {
                    'truth_prob': probabilities[:, 0],
                    'lie_prob': probabilities[:, 1],
                    'max_confidence': np.max(probabilities, axis=1)
                }
            
            return results
            
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")
    
    def predict_majority_vote(self, data):
        """
        Aggregate multiple frames into a single prediction using majority vote.
        
        Args:
            data: DataFrame with multiple frames (rows)
        
        Returns:
            dict: Single aggregated prediction
        """
        # Get individual predictions
        results = self.predict(data, return_probability=True)
        n_frames = len(results['predictions'])
        
        # Count votes
        lie_votes = sum(results['predictions'])
        truth_votes = n_frames - lie_votes
        
        final_prediction = 1 if lie_votes > truth_votes else 0
        final_label = 'lie' if final_prediction == 1 else 'truth'
        
        return {
            'aggregated_prediction': final_label,
            'aggregated_code': final_prediction,
            'method': 'majority_vote',
            'individual_predictions': results['labels'],
            'vote_counts': {'truth': truth_votes, 'lie': lie_votes},
            'total_frames': n_frames,
            'confidence_ratio': max(truth_votes, lie_votes) / n_frames
        }


if __name__ == "__main__":
    print("HRV Model Prediction Tool - Majority Vote Only")
    print("=" * 50)
    
    # Load the model (replace with your actual model filename)
    model_filename = "best_hrv_model_best_12feat.pkl"
    
    try:
        predictor = HRVPredictor(model_filename)
        
        # Load and predict from CSV using majority vote
        csv_file = "prediction_input.csv"  # Replace with your CSV file path
        
        result = predictor.predict_from_csv(csv_file)
        
        print(f"\nPrediction Result:")
        print(f"Final prediction: {result['aggregated_prediction']}")
        print(f"Vote counts: {result['vote_counts']}")
        print(f"Confidence ratio: {result['confidence_ratio']:.3f}")
        print(f"Total windows analyzed: {result['csv_info']['total_windows']}")
        
    except FileNotFoundError as e:
        if "model" in str(e):
            print(f"Model file '{model_filename}' not found!")
            print("Make sure you've run the training script first to create the PKL file.")
        else:
            print(f"CSV file not found: {e}")
            print("Make sure your CSV file exists and has the correct format.")
    except Exception as e:
        print(f"Error: {e}")