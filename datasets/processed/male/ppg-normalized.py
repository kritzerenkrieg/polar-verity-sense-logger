import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os
import glob
import sys

def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """
    Apply Butterworth bandpass filter to the signal
    """
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Ensure frequency values are valid
        if low <= 0 or high >= 1 or low >= high:
            raise ValueError(f"Invalid frequency range: {lowcut}-{highcut} Hz for fs={fs} Hz")
            
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    except Exception as e:
        print(f"Error in bandpass filter: {e}")
        return signal

def main():
    print("PPG Signal Processing and Normalization with 10-second Indicators")
    print("=" * 50)
    
    # Get current directory
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # If running in interactive mode
        current_dir = os.getcwd()
    
    print(f"Scanning directory: {current_dir}")
    
    # Find all .txt files recursively in subdirectories
    txt_files = glob.glob(os.path.join(current_dir, "**/*.txt"), recursive=True)
    
    # Also check for files in the current directory
    txt_files.extend(glob.glob(os.path.join(current_dir, "*.txt")))
    
    # Remove duplicates and sort alphabetically
    txt_files = sorted(list(set(txt_files)))
    
    if not txt_files:
        print("No .txt files found in current directory or subdirectories.")
        
        # Try alternative search patterns
        alternative_patterns = ["*.csv", "**/*.csv", "data/*", "datasets/*"]
        for pattern in alternative_patterns:
            alt_files = glob.glob(os.path.join(current_dir, pattern), recursive=True)
            if alt_files:
                print(f"Found files with pattern '{pattern}':")
                for f in alt_files[:10]:  # Show first 10 files
                    print(f"  - {f}")
                break
        return
    
    print(f"\nFound {len(txt_files)} .txt file(s) in directory and subdirectories:")
    for i, file in enumerate(txt_files):
        file_size = os.path.getsize(file) / 1024  # Size in KB
        rel_path = os.path.relpath(file, current_dir)  # Show relative path
        print(f"{i}: {rel_path} ({file_size:.1f} KB)")
    
    # File selection
    try:
        if len(txt_files) == 1:
            choice = 0
            print(f"Automatically selecting the only file: {os.path.basename(txt_files[0])}")
        else:
            choice = int(input(f"\nEnter file number (0-{len(txt_files)-1}): "))
            
        if choice < 0 or choice >= len(txt_files):
            raise IndexError("Invalid file selection.")
            
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        return
    
    file_path = txt_files[choice]
    print(f"\nProcessing file: {os.path.relpath(file_path, current_dir)}")
    print(f"Full path: {file_path}")
    
    try:
        # Read data with error handling
        print("Reading data...")
        try:
            # Try semicolon separator first
            df = pd.read_csv(file_path, sep=';', engine='python')
        except:
            # Fallback to comma separator
            df = pd.read_csv(file_path, sep=',', engine='python')
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Check for required columns
        required_channels = ['channel 0', 'channel 1', 'channel 2']
        available_channels = [col for col in required_channels if col in df.columns]
        
        if not available_channels:
            print("Warning: Standard channel columns not found. Available columns:")
            for col in df.columns:
                print(f"  - {col}")
            
            # Try to find numeric columns for PPG data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 1:
                print(f"Using numeric columns for PPG signal: {numeric_cols[:3]}")
                available_channels = numeric_cols[:3]
            else:
                print("No suitable numeric columns found for PPG signal.")
                return
        
        # Create PPG signal
        if len(available_channels) > 1:
            df['ppg_raw'] = df[available_channels].mean(axis=1)
            print(f"Created PPG signal from {len(available_channels)} channels")
        else:
            df['ppg_raw'] = df[available_channels[0]]
            print(f"Using single channel: {available_channels[0]}")
        
        # Handle timestamp
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
        if timestamp_cols:
            try:
                df['Phone timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
                time = (df['Phone timestamp'] - df['Phone timestamp'].iloc[0]).dt.total_seconds()
                print(f"Using timestamp column: {timestamp_cols[0]}")
            except:
                print("Could not parse timestamp, using sample indices")
                time = np.arange(len(df))
        else:
            print("No timestamp column found, using sample indices")
            time = np.arange(len(df))
        
        # Signal preprocessing
        print("Preprocessing signal...")
        ppg = df['ppg_raw']
        
        # Remove any NaN values
        valid_mask = ~np.isnan(ppg)
        ppg = ppg[valid_mask]
        time = time[valid_mask] if len(time) == len(valid_mask) else np.arange(len(ppg))
        
        if len(ppg) == 0:
            print("Error: No valid PPG data found.")
            return
        
        # Normalize signal
        ppg_centered = ppg - ppg.mean()
        ppg_scaled = ppg_centered / (ppg_centered.std() + 1e-10)  # Add small value to prevent division by zero
        
        # Apply bandpass filter
        print("Applying bandpass filter (0.5-5.0 Hz)...")
        lowcut = 0.5
        highcut = 5.0  # Set high cut-off frequency
        
        ppg_filtered = bandpass_filter(ppg_scaled.values, lowcut, highcut, fs=100, order=3)
        
        # Plot the normalized PPG signal
        print("Creating normalized PPG plot with 10-second indicators...")
        plt.figure(figsize=(12, 6))
        plt.plot(time, ppg_filtered, label='Normalized PPG Signal')

        # Add vertical lines every 10 seconds
        for t in range(0, int(np.max(time)), 10):
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Normalized PPG Amplitude')
        plt.title('Normalized PPG Signal with 10-Second Indicators')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
