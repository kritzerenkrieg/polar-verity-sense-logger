import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import os
import glob
import sys

def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if low <= 0 or high >= 1 or low >= high:
            raise ValueError(f"Invalid frequency range: {lowcut}-{highcut} Hz for fs={fs} Hz")
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    except Exception as e:
        print(f"Error in bandpass filter: {e}")
        return signal

def detect_artifacts_mad(nn_intervals, threshold=3.0):
    """Detect artifacts using Median Absolute Deviation (MAD)"""
    if len(nn_intervals) < 3:
        return np.zeros(len(nn_intervals), dtype=bool)
    
    median_nn = np.median(nn_intervals)
    mad = np.median(np.abs(nn_intervals - median_nn))
    
    # Avoid division by zero
    if mad == 0:
        mad = np.std(nn_intervals) * 0.6745  # Convert std to MAD equivalent
    
    # Calculate modified z-scores
    modified_z_scores = 0.6745 * (nn_intervals - median_nn) / mad
    
    # Mark as artifact if beyond threshold
    artifacts = np.abs(modified_z_scores) > threshold
    return artifacts

def detect_artifacts_percentage_change(nn_intervals, threshold=0.2):
    """Detect artifacts based on percentage change from previous interval"""
    if len(nn_intervals) < 2:
        return np.zeros(len(nn_intervals), dtype=bool)
    
    artifacts = np.zeros(len(nn_intervals), dtype=bool)
    
    for i in range(1, len(nn_intervals)):
        change = abs(nn_intervals[i] - nn_intervals[i-1]) / nn_intervals[i-1]
        if change > threshold:
            artifacts[i] = True
    
    return artifacts

def correct_artifacts(nn_intervals, nn_times, artifact_mask, method='interpolation'):
    """Correct detected artifacts"""
    if not np.any(artifact_mask):
        return nn_intervals, nn_times
    
    corrected_nn = nn_intervals.copy()
    
    if method == 'interpolation':
        # Only interpolate if we have enough good data points
        valid_indices = ~artifact_mask
        if np.sum(valid_indices) < 2:
            print("Warning: Too many artifacts for interpolation. Consider manual inspection.")
            return nn_intervals, nn_times
        
        valid_times = nn_times[valid_indices]
        valid_nn = nn_intervals[valid_indices]
        
        if len(valid_times) >= 2:
            # Use linear interpolation for artifacts
            interp_func = interp1d(valid_times, valid_nn, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            corrected_nn[artifact_mask] = interp_func(nn_times[artifact_mask])
    
    elif method == 'remove':
        # Remove artifacts entirely
        valid_mask = ~artifact_mask
        corrected_nn = nn_intervals[valid_mask]
        nn_times = nn_times[valid_mask]
    
    return corrected_nn, nn_times

def denoise_nn_intervals(nn_intervals, nn_times, mad_threshold=3.0, 
                        change_threshold=0.2, correction_method='interpolation'):
    """
    Comprehensive denoising of NN intervals
    """
    if len(nn_intervals) < 3:
        return nn_intervals, nn_times, np.array([])
    
    # Step 1: Detect artifacts using MAD
    artifacts_mad = detect_artifacts_mad(nn_intervals, mad_threshold)
    
    # Step 2: Detect artifacts based on percentage change
    artifacts_change = detect_artifacts_percentage_change(nn_intervals, change_threshold)
    
    # Combine artifact detection methods
    combined_artifacts = artifacts_mad | artifacts_change
    
    artifact_count = np.sum(combined_artifacts)
    artifact_percentage = (artifact_count / len(nn_intervals)) * 100
    
    # Step 3: Correct artifacts
    if artifact_percentage > 50:
        correction_method = 'remove'  # Force removal if too many artifacts
    
    corrected_nn, corrected_times = correct_artifacts(nn_intervals, nn_times, 
                                                     combined_artifacts, correction_method)
    
    return corrected_nn, corrected_times, combined_artifacts

def calculate_nn_intervals(peaks, time_array):
    if len(peaks) < 2:
        return np.array([]), np.array([])
    peak_times = time_array[peaks]
    nn_intervals = np.diff(peak_times) * 1000  # in ms
    
    # Apply physiological bounds
    valid_mask = (nn_intervals >= 300) & (nn_intervals <= 2000)
    nn_intervals_filtered = nn_intervals[valid_mask]
    peak_times_filtered = peak_times[1:][valid_mask]
    
    return nn_intervals_filtered, peak_times_filtered

def calculate_hrv_time_domain_metrics_short(nn_intervals):
    """Calculate HRV metrics suitable for short-term recordings (30 seconds)"""
    if len(nn_intervals) < 2:
        return None

    nn = np.array(nn_intervals)
    successive_diffs = np.diff(nn)
    
    # Basic statistics
    nn_count = len(nn)
    nn_mean = np.mean(nn)
    nn_min = np.min(nn)
    nn_max = np.max(nn)
    
    # Standard deviation of NN intervals
    sdnn = np.std(nn, ddof=1) if len(nn) > 1 else np.nan
    
    # Standard deviation of successive differences
    sdsd = np.std(successive_diffs, ddof=1) if len(successive_diffs) > 1 else np.nan
    
    # Root mean square of successive differences
    rmssd = np.sqrt(np.mean(successive_diffs**2)) if len(successive_diffs) > 0 else np.nan
    
    # Percentage of successive NN intervals differing by more than 20ms
    pnn20 = (np.sum(np.abs(successive_diffs) > 20) / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else np.nan
    
    # Percentage of successive NN intervals differing by more than 50ms
    pnn50 = (np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else np.nan
    
    # Triangular index (modified for short recordings)
    try:
        hist_counts = np.histogram(nn.astype(int), bins=range(int(nn_min), int(nn_max)+2))[0]
        triangular_index = len(nn) / np.max(hist_counts) if np.max(hist_counts) > 0 else np.nan
    except:
        triangular_index = np.nan

    return {
        'nn_count': nn_count,
        'nn_mean': nn_mean,
        'nn_min': nn_min,
        'nn_max': nn_max,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'rmssd': rmssd,
        'pnn20': pnn20,
        'pnn50': pnn50,
        'triangular_index': triangular_index
    }

def extract_window_nn_intervals(nn_intervals, nn_times, start_time, window_duration=30):
    """Extract NN intervals within a specific time window"""
    end_time = start_time + window_duration
    
    # Ensure inputs are numpy arrays
    nn_intervals = np.array(nn_intervals)
    nn_times = np.array(nn_times)
    
    # Find NN intervals within the time window
    window_mask = (nn_times >= start_time) & (nn_times < end_time)
    
    window_nn = nn_intervals[window_mask]
    window_times = nn_times[window_mask]
    
    return window_nn, window_times

def sliding_window_hrv_analysis_70_samples(nn_intervals, nn_times, target_samples=70):
    """
    Perform sliding window HRV analysis - EXACTLY 70 SAMPLES
    
    Parameters:
    - nn_intervals: array of NN intervals in ms
    - nn_times: array of NN interval timestamps in seconds
    - target_samples: exact number of samples to output (default: 70)
    
    Returns:
    - DataFrame with HRV metrics for exactly 70 windows
    """
    
    if len(nn_intervals) == 0 or len(nn_times) == 0:
        return pd.DataFrame()
    
    # Convert to numpy arrays to ensure proper indexing
    nn_intervals = np.array(nn_intervals)
    nn_times = np.array(nn_times)
    
    # Determine the total duration
    total_duration = nn_times[-1] - nn_times[0]
    
    # Calculate window parameters to get exactly 70 samples
    window_duration = 30  # Fixed window duration in seconds
    
    # Calculate step size to get exactly target_samples windows
    if target_samples <= 1:
        step_size = total_duration  # Single window
    else:
        step_size = (total_duration - window_duration) / (target_samples - 1)
    
    # Ensure step size is positive
    if step_size <= 0:
        step_size = 1.0
        # Adjust window duration if necessary
        if total_duration < window_duration:
            window_duration = total_duration * 0.8  # Use 80% of total duration
    
    # Calculate window start times for exactly target_samples windows
    window_starts = np.linspace(0, total_duration - window_duration, target_samples)
    
    print(f"Total recording duration: {total_duration:.1f} seconds")
    print(f"Window duration: {window_duration:.1f} seconds")
    print(f"Step size: {step_size:.3f} seconds")
    print(f"Target windows: {target_samples}")
    print(f"Calculated windows: {len(window_starts)}")
    
    results = []
    
    for i, start_time in enumerate(window_starts):
        # Extract NN intervals for this window
        window_nn, window_times = extract_window_nn_intervals(
            nn_intervals, nn_times, start_time, window_duration
        )
        
        # Calculate HRV metrics for this window
        if len(window_nn) >= 2:  # Need at least 2 NN intervals
            hrv_metrics = calculate_hrv_time_domain_metrics_short(window_nn)
            
            if hrv_metrics is not None:
                # Add window information
                hrv_metrics['window_start'] = start_time
                hrv_metrics['window_end'] = start_time + window_duration
                hrv_metrics['window_number'] = i + 1
                
                results.append(hrv_metrics)
            else:
                # Create empty entry for failed calculations
                empty_metrics = {
                    'nn_count': len(window_nn),
                    'nn_mean': np.nan,
                    'nn_min': np.nan,
                    'nn_max': np.nan,
                    'sdnn': np.nan,
                    'sdsd': np.nan,
                    'rmssd': np.nan,
                    'pnn20': np.nan,
                    'pnn50': np.nan,
                    'triangular_index': np.nan,
                    'window_start': start_time,
                    'window_end': start_time + window_duration,
                    'window_number': i + 1
                }
                results.append(empty_metrics)
        else:
            # Not enough data in this window
            empty_metrics = {
                'nn_count': len(window_nn),
                'nn_mean': np.nan,
                'nn_min': np.nan,
                'nn_max': np.nan,
                'sdnn': np.nan,
                'sdsd': np.nan,
                'rmssd': np.nan,
                'pnn20': np.nan,
                'pnn50': np.nan,
                'triangular_index': np.nan,
                'window_start': start_time,
                'window_end': start_time + window_duration,
                'window_number': i + 1
            }
            results.append(empty_metrics)
    
    df_results = pd.DataFrame(results)
    
    # Ensure we have exactly target_samples rows
    if len(df_results) > target_samples:
        df_results = df_results.head(target_samples)
    elif len(df_results) < target_samples:
        # Pad with empty rows if needed
        for i in range(len(df_results), target_samples):
            empty_row = {
                'nn_count': 0,
                'nn_mean': np.nan,
                'nn_min': np.nan,
                'nn_max': np.nan,
                'sdnn': np.nan,
                'sdsd': np.nan,
                'rmssd': np.nan,
                'pnn20': np.nan,
                'pnn50': np.nan,
                'triangular_index': np.nan,
                'window_start': np.nan,
                'window_end': np.nan,
                'window_number': i + 1
            }
            df_results = pd.concat([df_results, pd.DataFrame([empty_row])], ignore_index=True)
    
    print(f"Final output: {len(df_results)} windows (target: {target_samples})")
    
    return df_results

def highly_sensitive_peak_detection(signal, fs, min_hr=40, max_hr=180):
    min_distance = int(fs * 60 / max_hr)
    min_distance = max(min_distance, 1)
    prominence_threshold = 0.01 * np.std(signal)
    height_threshold = np.min(signal) - 1e-5

    peaks, properties = find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence_threshold,
        height=height_threshold,
        width=1
    )
    valid_peak_mask = signal[peaks] > 0
    peaks = peaks[valid_peak_mask]
    return peaks, properties

def process_single_file(file_path, show_plots=True):
    """Process a single PPG file and return HRV results - EXACTLY 70 SAMPLES"""
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    print("-" * 50)
    
    try:
        # Load and process data
        try:
            df = pd.read_csv(file_path, sep=';', engine='python')
        except:
            df = pd.read_csv(file_path, sep=',', engine='python')

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        required_channels = ['channel 0', 'channel 1', 'channel 2']
        available_channels = [col for col in required_channels if col in df.columns]

        if not available_channels:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 1:
                available_channels = numeric_cols[:3]
            else:
                print("No numeric channels found.")
                return None, file_path

        if len(available_channels) > 1:
            df['ppg_raw'] = df[available_channels].mean(axis=1)
        else:
            df['ppg_raw'] = df[available_channels[0]]

        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
        if timestamp_cols:
            try:
                df['Phone timestamp'] = pd.to_datetime(df[timestamp_cols[0]])
                time = (df['Phone timestamp'] - df['Phone timestamp'].iloc[0]).dt.total_seconds()
            except:
                time = np.arange(len(df))
        else:
            time = np.arange(len(df))

        ppg = df['ppg_raw']
        valid_mask = ~np.isnan(ppg)
        ppg = ppg[valid_mask]
        time = time[valid_mask] if len(time) == len(valid_mask) else np.arange(len(ppg))

        if len(ppg) == 0:
            print("Error: no valid PPG data.")
            return None, file_path

        ppg_centered = ppg - ppg.mean()
        ppg_scaled = ppg_centered / (ppg_centered.std() + 1e-10)

        if len(time) > 1:
            fs = 1 / np.median(np.diff(time))
            if fs > 1000 or fs < 1:
                fs = 100
                print("Using default sampling rate: 100 Hz")
            else:
                print(f"Estimated sampling frequency: {fs:.2f} Hz")
        else:
            fs = 100

        # Signal processing
        ppg_filtered = bandpass_filter(ppg_scaled.values, 0.5, min(5.0, fs/2.1), fs, order=3)
        peaks, properties = highly_sensitive_peak_detection(ppg_filtered, fs)
        print(f"Detected {len(peaks)} peaks")

        if len(peaks) > 1:
            nn_intervals_raw, nn_times_raw = calculate_nn_intervals(peaks, time)
            
            if len(nn_intervals_raw) > 0:
                print("Denoising NN intervals...")
                
                # Apply denoising
                nn_intervals, nn_times, artifacts = denoise_nn_intervals(
                    nn_intervals_raw, nn_times_raw,
                    mad_threshold=3.0,
                    change_threshold=0.2,
                    correction_method='interpolation'
                )
                
                print(f"Original NN intervals: {len(nn_intervals_raw)}")
                print(f"Final NN intervals after denoising: {len(nn_intervals)}")
                
                if len(nn_intervals) > 0:
                    print("Performing sliding window HRV analysis (70 samples)...")
                    
                    # Perform sliding window analysis - EXACTLY 70 SAMPLES
                    hrv_results = sliding_window_hrv_analysis_70_samples(
                        nn_intervals, nn_times, 
                        target_samples=70  # HARD-CODED to 70 samples
                    )
                    
                    if not hrv_results.empty:
                        print(f"Successfully analyzed {len(hrv_results)} windows")
                        
                        # Create visualization if requested
                        if show_plots:
                            create_sliding_window_plots(hrv_results, time, ppg_filtered, peaks, nn_intervals, nn_times)
                        
                        return hrv_results, file_path
                    else:
                        print("No valid windows found for analysis")
                        return None, file_path
                else:
                    print("No valid NN intervals after denoising")
                    return None, file_path
            else:
                print("No NN intervals detected")
                return None, file_path
        else:
            print("Insufficient peaks detected")
            return None, file_path

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None, file_path

def main():
    print("PPG Signal Processing - Sliding Window HRV Time Domain Analysis (70 Samples)")
    print("=" * 80)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    print(f"Scanning directory: {current_dir}")

    txt_files = glob.glob(os.path.join(current_dir, "**/*.txt"), recursive=True)
    txt_files.extend(glob.glob(os.path.join(current_dir, "*.txt")))
    txt_files = sorted(list(set(txt_files)))

    if not txt_files:
        print("No .txt files found.")
        return

    print(f"Found {len(txt_files)} .txt file(s):")
    for i, file in enumerate(txt_files):
        file_size = os.path.getsize(file) / 1024
        print(f"  {i}: {os.path.relpath(file, current_dir)} ({file_size:.1f} KB)")

    # Enhanced file selection
    print("\nProcessing Options:")
    print("0: Select and process a single file")
    print("1: Process all files")

    try:
        choice = int(input("Enter your choice (0 or 1): "))
        if choice not in [0, 1]:
            raise ValueError("Choice must be 0 or 1")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create output directory
    output_dir = os.path.join(current_dir, "datasets", "windowed")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if choice == 0:
        # Single file processing
        if len(txt_files) == 1:
            file_choice = 0
            print(f"Automatically selecting: {os.path.basename(txt_files[0])}")
        else:
            try:
                file_choice = int(input(f"Enter file number (0-{len(txt_files)-1}): "))
                if file_choice < 0 or file_choice >= len(txt_files):
                    raise IndexError("Invalid choice.")
            except (ValueError, IndexError) as e:
                print(f"Error: {e}")
                return

        file_path = txt_files[file_choice]
        hrv_results, processed_file = process_single_file(file_path, show_plots=True)
        
        if hrv_results is not None:
            # Save results
            base_name = os.path.splitext(os.path.basename(processed_file))[0]
            output_filename = os.path.join(output_dir, f"{base_name}_windowed.csv")
            hrv_results.to_csv(output_filename, index=False)
            print(f"\nResults saved to: {output_filename}")
            print(f"Output contains exactly {len(hrv_results)} samples")
            
            # Display summary statistics
            display_summary_statistics(hrv_results)
        else:
            print("Failed to process the selected file.")

    else:
        # Batch processing all files
        print(f"\nProcessing all {len(txt_files)} files...")
        successful_files = 0
        failed_files = 0
        
        for i, file_path in enumerate(txt_files):
            print(f"\n[{i+1}/{len(txt_files)}] Processing: {os.path.basename(file_path)}")
            
            hrv_results, processed_file = process_single_file(file_path, show_plots=False)
            
            if hrv_results is not None:
                # Save results
                base_name = os.path.splitext(os.path.basename(processed_file))[0]
                output_filename = os.path.join(output_dir, f"{base_name}_windowed.csv")
                hrv_results.to_csv(output_filename, index=False)
                print(f"✓ Results saved to: {os.path.relpath(output_filename, current_dir)} ({len(hrv_results)} samples)")
                successful_files += 1
            else:
                print(f"✗ Failed to process: {os.path.basename(processed_file)}")
                failed_files += 1
        
        print(f"\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Total files: {len(txt_files)}")
        print(f"Successfully processed: {successful_files}")
        print(f"Failed: {failed_files}")
        print(f"Output directory: {os.path.relpath(output_dir, current_dir)}")
        print(f"Each successful file contains exactly 70 samples")

def display_summary_statistics(hrv_results):
    """Display summary statistics for HRV results"""
    print("\nSUMMARY STATISTICS ACROSS ALL WINDOWS:")
    print("="*50)
    
    metrics_to_summarize = ['sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'nn_mean']
    for metric in metrics_to_summarize:
        if metric in hrv_results.columns:
            valid_values = hrv_results[metric].dropna()
            if len(valid_values) > 0:
                unit = "ms" if metric in ['sdnn', 'sdsd', 'rmssd', 'nn_mean'] else "%"
                print(f"{metric.upper()} ({unit}):")
                print(f"  Mean: {valid_values.mean():.2f}")
                print(f"  Std:  {valid_values.std():.2f}")
                print(f"  Min:  {valid_values.min():.2f}")
                print(f"  Max:  {valid_values.max():.2f}")
                print()

def create_sliding_window_plots(hrv_results, time, ppg_filtered, peaks, nn_intervals, nn_times):
    """Create comprehensive plots for sliding window analysis"""
    
    fig = plt.figure(figsize=(18, 16))
    
    # Plot 1: PPG Signal with peaks
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(time, ppg_filtered, label='Filtered PPG', color='blue', alpha=0.7)
    if len(peaks) > 0:
        plt.plot(time[peaks], ppg_filtered[peaks], 'ro', markersize=3, label=f'Peaks ({len(peaks)})')
    plt.title('PPG Signal with Detected Peaks')
    plt.ylabel('Normalized PPG')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: NN Intervals
    ax2 = plt.subplot(4, 2, 2)
    if len(nn_intervals) > 0:
        plt.plot(nn_times, nn_intervals, 'g-o', markersize=2, linewidth=1, label='NN Intervals')
        plt.axhline(y=np.mean(nn_intervals), color='red', linestyle='--', 
                   label=f"Mean: {np.mean(nn_intervals):.1f} ms")
        plt.ylim(300, 2000)
    plt.title('NN Intervals Over Time')
    plt.ylabel('NN Interval (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: SDNN over windows
    ax3 = plt.subplot(4, 2, 3)
    valid_sdnn = hrv_results.dropna(subset=['sdnn'])
    if not valid_sdnn.empty:
        plt.plot(valid_sdnn['window_start'], valid_sdnn['sdnn'], 'b-o', markersize=3, linewidth=1.5)
        plt.title('SDNN Over Time Windows')
        plt.ylabel('SDNN (ms)')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: SDSD over windows
    ax4 = plt.subplot(4, 2, 4)
    valid_sdsd = hrv_results.dropna(subset=['sdsd'])
    if not valid_sdsd.empty:
        plt.plot(valid_sdsd['window_start'], valid_sdsd['sdsd'], 'purple', marker='o', markersize=3, linewidth=1.5)
        plt.title('SDSD Over Time Windows')
        plt.ylabel('SDSD (ms)')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: RMSSD over windows
    ax5 = plt.subplot(4, 2, 5)
    valid_rmssd = hrv_results.dropna(subset=['rmssd'])
    if not valid_rmssd.empty:
        plt.plot(valid_rmssd['window_start'], valid_rmssd['rmssd'], 'r-o', markersize=3, linewidth=1.5)
        plt.title('RMSSD Over Time Windows')
        plt.ylabel('RMSSD (ms)')
        plt.grid(True, alpha=0.3)
    
    # Plot 6: pNN20 over windows
    ax6 = plt.subplot(4, 2, 6)
    valid_pnn20 = hrv_results.dropna(subset=['pnn20'])
    if not valid_pnn20.empty:
        plt.plot(valid_pnn20['window_start'], valid_pnn20['pnn20'], 'orange', marker='o', markersize=3, linewidth=1.5)
        plt.title('pNN20 Over Time Windows')
        plt.ylabel('pNN20 (%)')
        plt.grid(True, alpha=0.3)
    
    # Plot 7: pNN50 over windows
    ax7 = plt.subplot(4, 2, 7)
    valid_pnn50 = hrv_results.dropna(subset=['pnn50'])
    if not valid_pnn50.empty:
        plt.plot(valid_pnn50['window_start'], valid_pnn50['pnn50'], 'g-o', markersize=3, linewidth=1.5)
        plt.title('pNN50 Over Time Windows')
        plt.ylabel('pNN50 (%)')
        plt.grid(True, alpha=0.3)
    
    # Plot 8: Mean Heart Rate over windows
    ax8 = plt.subplot(4, 2, 8)
    valid_hr = hrv_results.dropna(subset=['nn_mean'])
    if not valid_hr.empty:
        # Convert mean NN interval to heart rate (BPM)
        heart_rate = 60000 / valid_hr['nn_mean']  # 60000 ms per minute
        plt.plot(valid_hr['window_start'], heart_rate, 'm-o', markersize=3, linewidth=1.5)
        plt.title('Heart Rate Over Time Windows')
        plt.ylabel('Heart Rate (BPM)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()