import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
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

def calculate_nn_intervals(peaks, time_array):
    """
    Calculate NN intervals (Normal-to-Normal intervals) from detected peaks
    NN intervals are the time differences between consecutive normal heartbeats
    """
    if len(peaks) < 2:
        return np.array([]), np.array([])
    
    # Convert peak indices to time
    peak_times = time_array[peaks]
    
    # Calculate NN intervals in milliseconds
    nn_intervals = np.diff(peak_times) * 1000  # Convert to milliseconds
    
    # Filter out unrealistic intervals (outside physiological range)
    # Typical human NN intervals: 300ms (200 BPM) to 2000ms (30 BPM)
    valid_mask = (nn_intervals >= 300) & (nn_intervals <= 2000)
    nn_intervals_filtered = nn_intervals[valid_mask]
    peak_times_filtered = peak_times[1:][valid_mask]  # Corresponding peak times
    
    return nn_intervals_filtered, peak_times_filtered

def calculate_hrv_time_domain_metrics(nn_intervals):
    """
    Calculate comprehensive HRV time-domain metrics for research
    
    Parameters:
    nn_intervals: array of NN intervals in milliseconds
    
    Returns:
    dict: Dictionary containing all HRV metrics
    """
    if len(nn_intervals) < 2:
        return None
    
    # Convert to numpy array for calculations
    nn = np.array(nn_intervals)
    
    # Calculate successive differences
    successive_diffs = np.diff(nn)
    
    # ===== DEVIATION-BASED APPROACH =====
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(nn, ddof=1)
    
    # For SDANN and SDNN Index, we need to segment the data into 5-minute windows
    # Assuming typical recording, we'll estimate based on mean NN interval
    mean_nn = np.mean(nn)
    approx_beats_per_5min = int(5 * 60 * 1000 / mean_nn)  # Approximate beats in 5 minutes
    
    # SDANN: Standard deviation of average NN intervals for each 5 min segment
    if len(nn) >= approx_beats_per_5min * 2:  # Need at least 2 segments
        segments = []
        for i in range(0, len(nn), approx_beats_per_5min):
            segment = nn[i:i + approx_beats_per_5min]
            if len(segment) >= 10:  # Minimum beats per segment
                segments.append(np.mean(segment))
        
        if len(segments) >= 2:
            sdann = np.std(segments, ddof=1)
        else:
            sdann = np.nan
    else:
        sdann = np.nan
    
    # SDNN Index: Mean of the standard deviations of NN intervals in 5 min segments
    if len(nn) >= approx_beats_per_5min * 2:
        segment_stds = []
        for i in range(0, len(nn), approx_beats_per_5min):
            segment = nn[i:i + approx_beats_per_5min]
            if len(segment) >= 10:  # Minimum beats per segment
                segment_stds.append(np.std(segment, ddof=1))
        
        if len(segment_stds) >= 2:
            sdnn_index = np.mean(segment_stds)
        else:
            sdnn_index = np.nan
    else:
        sdnn_index = np.nan
    
    # ===== DIFFERENCE-BASED APPROACH =====
    
    # SDSD: Standard deviation of successive NN interval differences
    sdsd = np.std(successive_diffs, ddof=1)
    
    # RMSSD: Root mean square of successive NN interval differences
    rmssd = np.sqrt(np.mean(successive_diffs**2))
    
    # pNN20: Proportion of successive NN interval differences larger than 20 ms
    pnn20 = (np.sum(np.abs(successive_diffs) > 20) / len(successive_diffs)) * 100
    
    # pNN50: Proportion of successive NN interval differences larger than 50 ms
    pnn50 = (np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs)) * 100
    
    # Additional useful metrics
    nn_mean = np.mean(nn)
    nn_min = np.min(nn)
    nn_max = np.max(nn)
    
    # Package results
    hrv_metrics = {
        # Basic statistics
        'nn_count': len(nn),
        'nn_mean': nn_mean,
        'nn_min': nn_min,
        'nn_max': nn_max,
        
        # Deviation-based approach
        'sdnn': sdnn,
        'sdann': sdann,
        'sdnn_index': sdnn_index,
        
        # Difference-based approach
        'sdsd': sdsd,
        'rmssd': rmssd,
        'pnn20': pnn20,
        'pnn50': pnn50,
        
        # Additional derived metrics
        'triangular_index': len(nn) / np.max(np.histogram(nn.astype(int), bins=range(int(nn_min), int(nn_max)+1))[0]) if len(nn) > 0 else np.nan
    }
    
    return hrv_metrics

def highly_sensitive_peak_detection(signal, fs, min_hr=40, max_hr=180):
    """
    Peak detection with minimal filtering to capture all small spikes.
    """
    # No smoothing applied to maximize sensitivity
    smoothed_signal = signal

    # Set minimum distance based on max HR (allowing closely spaced peaks)
    min_distance = int(fs * 60 / max_hr)
    if min_distance < 1:
        min_distance = 1  # At least 1 sample apart

    # Very low thresholds to detect all spikes
    prominence_threshold = 0.01 * np.std(smoothed_signal)  # Very low prominence
    height_threshold = np.min(smoothed_signal) - 1e-5  # Allow all peaks above minimum signal

    peaks, properties = find_peaks(
        smoothed_signal,
        distance=min_distance,
        prominence=prominence_threshold,
        height=height_threshold,
        width=1  # Minimum width
    )
    
    # Filter out peaks below zero amplitude
    valid_peak_mask = smoothed_signal[peaks] > 0
    peaks = peaks[valid_peak_mask]

    return peaks, properties

def main():
    print("PPG Signal Processing and Heart Rate Analysis")
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
        
        # Estimate sampling frequency
        if len(time) > 1:
            fs = 1 / np.median(np.diff(time))
            if fs > 1000 or fs < 1:  # Unrealistic sampling rate
                fs = 100  # Default assumption
                print(f"Warning: Calculated sampling rate seems unrealistic. Using default: {fs} Hz")
            else:
                print(f"Estimated sampling frequency: {fs:.2f} Hz")
        else:
            fs = 100
            print(f"Using default sampling frequency: {fs} Hz")
        
        # Apply bandpass filter
        print("Applying bandpass filter (0.5-5.0 Hz)...")
        lowcut = 0.5
        highcut = min(5.0, fs/2.1)  # Ensure highcut is below Nyquist frequency
        
        ppg_filtered = bandpass_filter(ppg_scaled.values, lowcut, highcut, fs, order=3)
        
        # Peak detection with high sensitivity
        print("Detecting peaks with high sensitivity...")
        peaks, properties = highly_sensitive_peak_detection(ppg_filtered, fs)
        
        print(f"Detected {len(peaks)} peaks")
        
        if len(peaks) > 1:
            # Calculate NN intervals
            nn_intervals, nn_times = calculate_nn_intervals(peaks, time)
            if len(nn_intervals) > 0:
                # Calculate comprehensive HRV metrics
                hrv_metrics = calculate_hrv_time_domain_metrics(nn_intervals)
                
                if hrv_metrics:
                    print(f"\n{'='*60}")
                    print(f"HRV TIME-DOMAIN ANALYSIS RESULTS")
                    print(f"{'='*60}")
                    
                    print(f"\nBasic NN Interval Statistics:")
                    print(f"  • Number of NN intervals: {hrv_metrics['nn_count']}")
                    print(f"  • Mean NN interval: {hrv_metrics['nn_mean']:.1f} ms")
                    print(f"  • Min NN interval: {hrv_metrics['nn_min']:.1f} ms") 
                    print(f"  • Max NN interval: {hrv_metrics['nn_max']:.1f} ms")
                    
                    print(f"\nDEVIATION-BASED APPROACH:")
                    print(f"  • SDNN (Standard deviation of NN intervals): {hrv_metrics['sdnn']:.2f} ms")
                    if not np.isnan(hrv_metrics['sdann']):
                        print(f"  • SDANN (SD of 5-min NN averages): {hrv_metrics['sdann']:.2f} ms")
                    else:
                        print(f"  • SDANN: Not enough data for 5-min segments")
                    
                    if not np.isnan(hrv_metrics['sdnn_index']):
                        print(f"  • SDNN Index (Mean of 5-min NN SDs): {hrv_metrics['sdnn_index']:.2f} ms")
                    else:
                        print(f"  • SDNN Index: Not enough data for 5-min segments")
                    
                    print(f"\nDIFFERENCE-BASED APPROACH:")
                    print(f"  • SDSD (SD of successive differences): {hrv_metrics['sdsd']:.2f} ms")
                    print(f"  • RMSSD (RMS of successive differences): {hrv_metrics['rmssd']:.2f} ms")
                    print(f"  • pNN20 (% of |ΔNN| > 20ms): {hrv_metrics['pnn20']:.2f}%")
                    print(f"  • pNN50 (% of |ΔNN| > 50ms): {hrv_metrics['pnn50']:.2f}%")
                    
                    print(f"\nAdditional Metric:")
                    print(f"  • Triangular Index: {hrv_metrics['triangular_index']:.2f}")
                    
                    print(f"{'='*60}")
                else:
                    print("Error: Could not calculate HRV metrics")
        
        # Plotting
        print("Creating plots...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Filtered signal with peaks
        ax1.plot(time, ppg_filtered, label='Filtered PPG Signal', color='blue', linewidth=1)
        if len(peaks) > 0:
            ax1.plot(time[peaks], ppg_filtered[peaks], 'ro', markersize=4, label=f'Detected Peaks ({len(peaks)})')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Normalized PPG Signal')
        ax1.set_title('PPG Signal Analysis with Peak Detection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: NN intervals over time
        if len(peaks) > 1 and 'nn_times' in locals() and len(nn_intervals) > 0:
            ax2.plot(nn_times, nn_intervals, 'g-o', markersize=3, linewidth=1.5)
            ax2.axhline(y=np.mean(nn_intervals), color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean: {np.mean(nn_intervals):.1f} ms')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('NN Interval (ms)')
            ax2.set_title('NN Intervals (Normal-to-Normal Intervals)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(300, 2000)  # Physiological NN interval range
        else:
            ax2.text(0.5, 0.5, 'Insufficient peaks for NN interval calculation', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('NN Interval Analysis - Insufficient Data')
        
        # Plot 3: NN interval histogram for HRV visualization
        if len(peaks) > 1 and 'nn_intervals' in locals() and len(nn_intervals) > 0:
            ax3.hist(nn_intervals, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(x=np.mean(nn_intervals), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(nn_intervals):.1f} ms')
            ax3.axvline(x=np.mean(nn_intervals) + np.std(nn_intervals), color='orange', 
                       linestyle=':', label=f'Mean + SD: {np.mean(nn_intervals) + np.std(nn_intervals):.1f} ms')
            ax3.axvline(x=np.mean(nn_intervals) - np.std(nn_intervals), color='orange', 
                       linestyle=':', label=f'Mean - SD: {np.mean(nn_intervals) - np.std(nn_intervals):.1f} ms')
            ax3.set_xlabel('NN Interval (ms)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('NN Interval Distribution (HRV Histogram)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for histogram', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('NN Interval Distribution - Insufficient Data')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
