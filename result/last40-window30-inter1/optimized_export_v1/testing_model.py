#!/usr/bin/env python3
"""
HRV Lie Detection Model Loader - FILE INPUT VERSION
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
    
    def get_column_schema(self):
        """
        Get the expected column schema for input data files.
        
        Returns:
            dict: Schema information for time and frequency domain files
        """
        schema = {
            'time_domain_columns': {
                'required_features': self.feature_domains['time_features'],
                'optional_metadata': ['subject', 'condition', 'label', 'binary_label', 'window_number'],
                'description': 'Time domain HRV features file (CSV format)'
            },
            'frequency_domain_columns': {
                'required_features': self.feature_domains['freq_features'],
                'optional_metadata': ['subject', 'condition', 'label', 'binary_label', 'window_number'],
                'description': 'Frequency domain HRV features file (CSV format)'
            },
            'feature_descriptions': {
                # Time domain features
                'nn_mean': 'Mean of NN intervals (ms)',
                'nn_min': 'Minimum NN interval (ms)',
                'nn_count': 'Number of NN intervals',
                'sdnn': 'Standard deviation of NN intervals (ms)',
                'sdsd': 'Standard deviation of successive differences (ms)',
                'rmssd': 'Root mean square of successive differences (ms)',
                'pnn20': 'Percentage of NN intervals > 20ms different from previous (%, 0-100)',
                'pnn50': 'Percentage of NN intervals > 50ms different from previous (%, 0-100)',
                
                # Frequency domain features
                'lf_power': 'Low frequency power (0.04-0.15 Hz) in ms²',
                'hf_power': 'High frequency power (0.15-0.4 Hz) in ms²',
                'lf_norm': 'Normalized LF power (%, 0-100)',
                'hf_norm': 'Normalized HF power (%, 0-100)',
                'lf_hf_ratio': 'Ratio of LF to HF power',
                'ln_hf': 'Natural logarithm of HF power'
            }
        }
        return schema
    
    def print_schema(self):
        """Print the expected column schema in a readable format."""
        schema = self.get_column_schema()
        
        print("\n" + "="*60)
        print("EXPECTED DATA FILE SCHEMA")
        print("="*60)
        
        if schema['time_domain_columns']['required_features']:
            print("\nTIME DOMAIN FILE (e.g., 'time_features.csv'):")
            print("-" * 50)
            print("Required columns:")
            for feature in schema['time_domain_columns']['required_features']:
                desc = schema['feature_descriptions'].get(feature, 'No description available')
                print(f"  - {feature}: {desc}")
        
        if schema['frequency_domain_columns']['required_features']:
            print("\nFREQUENCY DOMAIN FILE (e.g., 'freq_features.csv'):")
            print("-" * 50)
            print("Required columns:")
            for feature in schema['frequency_domain_columns']['required_features']:
                desc = schema['feature_descriptions'].get(feature, 'No description available')
                print(f"  - {feature}: {desc}")
        
        print("\nOPTIONAL METADATA COLUMNS (for both files):")
        print("-" * 50)
        for col in schema['time_domain_columns']['optional_metadata']:
            print(f"  - {col}")
        
        print("\nNOTES:")
        print("-" * 50)
        print("- Files should be in CSV format")
        print("- Each row represents one measurement/window")
        print("- If both time and frequency files have metadata columns,")
        print("  they will be used to merge the data properly")
        print("- If no metadata columns, files must have same number of rows")
        print("  in the same order")
        print("="*60)
    
    def load_data_files(self, time_file=None, freq_file=None):
        """
        Load HRV data from CSV files.
        
        Args:
            time_file (str): Path to time domain features CSV file
            freq_file (str): Path to frequency domain features CSV file
            
        Returns:
            tuple: (df_time, df_frequency) DataFrames
        """
        df_time = None
        df_frequency = None
        
        # Load time domain data
        if time_file:
            if not os.path.exists(time_file):
                raise FileNotFoundError(f"Time domain file not found: {time_file}")
            
            print(f"Loading time domain data from: {time_file}")
            df_time = pd.read_csv(time_file)
            print(f"Time domain data shape: {df_time.shape}")
            
            # Check required features
            missing_features = [f for f in self.feature_domains['time_features'] 
                              if f not in df_time.columns]
            if missing_features:
                raise ValueError(f"Missing required time domain features: {missing_features}")
        
        # Load frequency domain data
        if freq_file:
            if not os.path.exists(freq_file):
                raise FileNotFoundError(f"Frequency domain file not found: {freq_file}")
            
            print(f"Loading frequency domain data from: {freq_file}")
            df_frequency = pd.read_csv(freq_file)
            print(f"Frequency domain data shape: {df_frequency.shape}")
            
            # Check required features
            missing_features = [f for f in self.feature_domains['freq_features'] 
                              if f not in df_frequency.columns]
            if missing_features:
                raise ValueError(f"Missing required frequency domain features: {missing_features}")
        
        return df_time, df_frequency
    
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
    
    def predict_from_files(self, time_file=None, freq_file=None, return_probabilities=False):
        """
        Make predictions directly from CSV files.
        
        Args:
            time_file (str): Path to time domain features CSV file
            freq_file (str): Path to frequency domain features CSV file
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            array: Predictions (0=truth, 1=lie) and optionally probabilities
        """
        # Load data from files
        df_time, df_frequency = self.load_data_files(time_file, freq_file)
        
        # Make predictions
        return self.predict(df_time, df_frequency, return_probabilities)
    
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

def demo_with_real_data(time_file=None, freq_file=None):
    """
    Demonstrate the model with real HRV data files.
    
    Args:
        time_file (str): Path to time domain CSV file
        freq_file (str): Path to frequency domain CSV file
    """
    print("\n=== HRV Lie Detection Model - Real Data Demo ===")
    
    try:
        # Load the trained model
        detector = HRVLieDetectorLoader()
        
        # Show expected schema
        detector.print_schema()
        
        # If no files provided, list available CSV files
        if not time_file and not freq_file:
            print("\nLooking for CSV files in current directory...")
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                print("Available CSV files:")
                for i, file in enumerate(csv_files, 1):
                    print(f"  {i}. {file}")
                print("\nTo use this demo, specify file paths:")
                print("  python script.py time_features.csv freq_features.csv")
                print("  OR modify the demo_with_real_data() call at the bottom")
            else:
                print("No CSV files found in current directory.")
            return
        
        # Load and make predictions
        print(f"\nLoading data files...")
        print(f"Time domain file: {time_file}")
        print(f"Frequency domain file: {freq_file}")
        
        predictions, probabilities = detector.predict_from_files(
            time_file, freq_file, return_probabilities=True
        )
        
        print(f"\nPredictions completed for {len(predictions)} samples")
        print("\nResults:")
        print("-" * 50)
        
        # Show first 10 predictions
        max_show = min(10, len(predictions))
        for i in range(max_show):
            pred_label = "LIE" if predictions[i] == 1 else "TRUTH"
            confidence = max(probabilities[i])
            print(f"Sample {i+1}: {pred_label} (confidence: {confidence:.3f})")
        
        if len(predictions) > 10:
            print(f"... and {len(predictions) - 10} more samples")
        
        # Summary statistics
        n_lies = sum(predictions)
        n_truths = len(predictions) - n_lies
        avg_confidence = np.mean([max(prob) for prob in probabilities])
        
        print(f"\nSummary:")
        print(f"Total samples: {len(predictions)}")
        print(f"Predicted lies: {n_lies} ({n_lies/len(predictions)*100:.1f}%)")
        print(f"Predicted truths: {n_truths} ({n_truths/len(predictions)*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Show model information
        print("\n=== Model Information ===")
        info = detector.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
        
        print("\n=== Demo Complete ===")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

# Example usage functions
def create_example_csv_files():
    """Create example CSV files with the correct schema for testing."""
    print("Creating example CSV files...")
    
    # Create example time domain data
    time_data = {
        'subject': ['S001', 'S001', 'S002', 'S002', 'S003'],
        'condition': ['baseline', 'test', 'baseline', 'test', 'baseline'],
        'window_number': [1, 1, 1, 1, 1],
        'nn_mean': [850, 820, 900, 880, 860],
        'nn_min': [650, 620, 700, 680, 660],
        'nn_count': [30, 32, 28, 30, 31],
        'sdnn': [45, 52, 38, 48, 42],
        'sdsd': [35, 42, 28, 38, 33],
        'rmssd': [40, 48, 32, 42, 37],
        'pnn20': [15, 18, 12, 16, 14],
        'pnn50': [8, 12, 6, 10, 7]
    }
    
    # Create example frequency domain data
    freq_data = {
        'subject': ['S001', 'S001', 'S002', 'S002', 'S003'],
        'condition': ['baseline', 'test', 'baseline', 'test', 'baseline'],
        'window_number': [1, 1, 1, 1, 1],
        'lf_power': [1200, 1100, 1400, 1300, 1250],
        'hf_power': [800, 750, 900, 850, 820],
        'lf_norm': [60, 59, 61, 60, 60],
        'hf_norm': [40, 41, 39, 40, 40],
        'lf_hf_ratio': [1.5, 1.47, 1.56, 1.53, 1.52],
        'ln_hf': [6.5, 6.4, 6.6, 6.55, 6.48]
    }
    
    # Save to CSV files
    pd.DataFrame(time_data).to_csv('example_time_features.csv', index=False)
    pd.DataFrame(freq_data).to_csv('example_freq_features.csv', index=False)
    
    print("Created example files:")
    print("- example_time_features.csv")
    print("- example_freq_features.csv")
    print("\nYou can now run the demo with these files!")

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        # Files provided as command line arguments
        time_file = sys.argv[1]
        freq_file = sys.argv[2]
        demo_with_real_data(time_file, freq_file)
    elif len(sys.argv) == 2 and sys.argv[1] == "create_example":
        # Create example files
        create_example_csv_files()
    else:
        # No files provided - show schema and available files
        print("Usage options:")
        print("1. python script.py time_features.csv freq_features.csv")
        print("2. python script.py create_example  # Creates example CSV files")
        print("3. python script.py  # Shows schema and available files")
        print()
        
        # Show schema and available files
        try:
            detector = HRVLieDetectorLoader()
            demo_with_real_data()  # This will show schema and available files
        except FileNotFoundError:
            print("Model file 'best_hrv_model_best_12feat.pkl' not found!")
            print("Make sure the model file is in the same directory.")