#!/usr/bin/env python3
"""
PPG Signal Processing for HRV Lie Detection Analysis
Processes raw PPG data to generate consistent windowed features for both time and frequency domains.
Ensures same number of windows per subject-condition combination.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch, periodogram
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import trapz
import os
import glob
import sys
import re

def parse_filename(filename):
    """
    Parse filename to extract metadata.
    Expected format: pr_{subject}_{condition}_{truth|lie}-sequence.txt
    """
    base_name = os.path.splitext(filename)[0]
    pattern = r'pr_([^_]+)_([^_]+)_(truth|lie)-sequence'
    match = re.match(pattern, base_name)
    
    if match:
        return {
            'subject': match.group(1),
            'condition': match.group(2),
            'label': match.group(3)
        }
    return None

def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """Apply bandpass filter to signal"""
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
    """Comprehensive denoising of NN intervals"""
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
    """Calculate NN intervals from detected peaks"""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    peak_times = time_array[peaks]
    nn_intervals = np.diff(peak_times) * 1000  # in ms
    
    # Apply physiological bounds
    valid_mask = (nn_intervals >= 300) & (nn_intervals <= 2000)
    nn_intervals_filtered = nn_intervals[valid_mask]
    peak_times_filtered = peak_times[1:][valid_mask]
    
    return nn_intervals_filtered, peak_times_filtered

def highly_sensitive_peak_detection(signal, fs, min_hr=40, max_hr=180):
    """Detect peaks in PPG signal"""
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

def calculate_hrv_time_domain_metrics_short(nn_intervals):
    """Calculate HRV time domain metrics suitable for short-term recordings (30 seconds)"""
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

def interpolate_nn_intervals(nn_intervals, nn_times, fs_target=4.0):
    """Interpolate NN intervals to create evenly sampled signal for frequency analysis"""
    if len(nn_intervals) < 3:
        return None, None
    
    # Create time vector for target sampling rate
    duration = nn_times[-1] - nn_times[0]
    time_interp = np.arange(nn_times[0], nn_times[-1], 1/fs_target)
    
    # Use cubic spline interpolation for smoother results
    try:
        # Ensure we have enough points for cubic spline
        if len(nn_intervals) >= 4:
            spline = UnivariateSpline(nn_times, nn_intervals, s=0)
            nn_interp = spline(time_interp)
        else:
            # Fall back to linear interpolation
            interp_func = interp1d(nn_times, nn_intervals, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            nn_interp = interp_func(time_interp)
        
        return nn_interp, time_interp
    except Exception as e:
        print(f"Interpolation error: {e}")
        return None, None

def calculate_psd_welch(nn_interp, fs, nperseg=None):
    """Calculate Power Spectral Density using Welch's method"""
    if nperseg is None:
        # Use segment length that gives good frequency resolution
        nperseg = min(len(nn_interp) // 2, int(fs * 60))  # Max 60 seconds or half the signal
        nperseg = max(nperseg, int(fs * 10))  # Min 10 seconds
    
    try:
        # Remove mean (detrend)
        nn_detrended = nn_interp - np.mean(nn_interp)
        
        # Calculate noverlap (50% overlap)
        noverlap = int(nperseg * 0.5)
        
        # Apply window and calculate PSD
        freq, psd = welch(nn_detrended, fs=fs, nperseg=nperseg, 
                         window='hann', noverlap=noverlap, detrend='linear')
        
        return freq, psd
    except Exception as e:
        print(f"PSD calculation error: {e}")
        return None, None

def calculate_hrv_frequency_domain_metrics(nn_intervals, nn_times, method='welch'):
    """Calculate HRV frequency domain metrics"""
    if len(nn_intervals) < 10:  # Need sufficient data for frequency analysis
        return None
    
    # Interpolate NN intervals
    fs_target = 4.0  # 4 Hz is standard for HRV analysis
    nn_interp, time_interp = interpolate_nn_intervals(nn_intervals, nn_times, fs_target)
    
    if nn_interp is None:
        return None
    
    # Calculate Power Spectral Density
    if method == 'welch':
        freq, psd = calculate_psd_welch(nn_interp, fs_target)
    else:
        # Periodogram method
        try:
            nn_detrended = nn_interp - np.mean(nn_interp)
            freq, psd = periodogram(nn_detrended, fs=fs_target, window='hann', detrend='linear')
        except Exception as e:
            print(f"Periodogram calculation error: {e}")
            return None
    
    if freq is None or psd is None:
        return None
    
    # Define frequency bands (in Hz)
    lf_band = (0.04, 0.15)      # Low Frequency
    hf_band = (0.15, 0.4)       # High Frequency
    
    # Find frequency indices for each band
    lf_idx = (freq >= lf_band[0]) & (freq < lf_band[1])
    hf_idx = (freq >= hf_band[0]) & (freq < hf_band[1])
    
    # Calculate power in each band using trapezoidal integration
    lf_power = trapz(psd[lf_idx], freq[lf_idx]) if np.any(lf_idx) else 0
    hf_power = trapz(psd[hf_idx], freq[hf_idx]) if np.any(hf_idx) else 0
    
    # Calculate derived metrics
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else np.nan
    
    # Normalized powers (relative to LF + HF total)
    lf_plus_hf_total = lf_power + hf_power
    lf_norm = (lf_power / lf_plus_hf_total) * 100 if lf_plus_hf_total > 0 else np.nan
    hf_norm = (hf_power / lf_plus_hf_total) * 100 if lf_plus_hf_total > 0 else np.nan
    
    # Natural logarithm of HF power
    ln_hf = np.log(hf_power) if hf_power > 0 else np.nan
    
    # Peak frequencies (frequency with maximum power in each band)
    lf_peak = freq[lf_idx][np.argmax(psd[lf_idx])] if np.any(lf_idx) and np.sum(lf_idx) > 0 else np.nan
    hf_peak = freq[hf_idx][np.argmax(psd[hf_idx])] if np.any(hf_idx) and np.sum(hf_idx) > 0 else np.nan
    
    return {
        'lf_power': lf_power,        
        'hf_power': hf_power,        
        'lf_hf_ratio': lf_hf_ratio,  
        'lf_norm': lf_norm,          
        'hf_norm': hf_norm,          
        'ln_hf': ln_hf,              
        'lf_peak': lf_peak,          
        'hf_peak': hf_peak,          
        'n_samples': len(nn_interp)  
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

def sliding_window_hrv_analysis_last40_seconds(nn_intervals, nn_times, domain='time'):
    """
    Perform sliding window HRV analysis on LAST 40 SECONDS ONLY
    30-second windows, moving 1 second forward each time = 11 windows total
    
    Parameters:
    - nn_intervals: array of NN intervals in ms
    - nn_times: array of NN interval timestamps in seconds
    - domain: 'time' or 'frequency'
    
    Returns:
    - DataFrame with HRV metrics for exactly 11 windows (last 40 seconds)
    """
    
    if len(nn_intervals) == 0 or len(nn_times) == 0:
        return pd.DataFrame()
    
    # Convert to numpy arrays to ensure proper indexing
    nn_intervals = np.array(nn_intervals)
    nn_times = np.array(nn_times)
    
    # Determine the total duration
    total_duration = nn_times[-1] - nn_times[0]
    
    # FIXED PARAMETERS FOR LAST 40 SECONDS ANALYSIS
    analysis_start_time = total_duration - 40  # Start 40 seconds from the end
    analysis_end_time = total_duration        # End at the very end
    window_duration = 30                      # 30-second windows
    step_size = 1                            # Move 1 second forward each time
    
    # Calculate number of windows: (40 - 30) / 1 + 1 = 11 windows
    # Window 1: [60-90], Window 2: [61-91], ..., Window 11: [70-100]
    n_windows = int((analysis_end_time - analysis_start_time - window_duration) / step_size) + 1
    
    # Generate window start times
    window_starts = []
    for i in range(n_windows):
        start_time = analysis_start_time + (i * step_size)
        if start_time + window_duration <= analysis_end_time:
            window_starts.append(start_time)
    
    print(f"Total recording duration: {total_duration:.1f} seconds")
    print(f"Analysis period: last 40 seconds ({analysis_start_time:.1f}-{analysis_end_time:.1f}s)")
    print(f"Window duration: {window_duration} seconds")
    print(f"Step size: {step_size} second")
    print(f"Number of windows: {len(window_starts)}")
    print(f"Window times: {[f'{s:.0f}-{s+window_duration:.0f}s' for s in window_starts[:3]]} ... {[f'{s:.0f}-{s+window_duration:.0f}s' for s in window_starts[-2:]]}")
    
    results = []
    
    for i, start_time in enumerate(window_starts):
        # Extract NN intervals for this window
        window_nn, window_times = extract_window_nn_intervals(
            nn_intervals, nn_times, start_time, window_duration
        )
        
        if domain == 'time':
            # Calculate HRV time domain metrics for this window
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
        
        elif domain == 'frequency':
            # Calculate HRV frequency domain metrics for this window
            if len(window_nn) >= 10:  # Need at least 10 NN intervals for frequency analysis
                hrv_metrics = calculate_hrv_frequency_domain_metrics(window_nn, window_times)
                
                if hrv_metrics is not None:
                    # Add window information
                    hrv_metrics['window_start'] = start_time
                    hrv_metrics['window_end'] = start_time + window_duration
                    hrv_metrics['window_number'] = i + 1
                    hrv_metrics['nn_count'] = len(window_nn)
                    
                    results.append(hrv_metrics)
                else:
                    # Create empty entry for failed calculations
                    empty_metrics = {
                        'lf_power': np.nan,
                        'hf_power': np.nan,
                        'lf_hf_ratio': np.nan,
                        'lf_norm': np.nan,
                        'hf_norm': np.nan,
                        'ln_hf': np.nan,
                        'lf_peak': np.nan,
                        'hf_peak': np.nan,
                        'n_samples': 0,
                        'window_start': start_time,
                        'window_end': start_time + window_duration,
                        'window_number': i + 1,
                        'nn_count': len(window_nn)
                    }
                    results.append(empty_metrics)
            else:
                # Not enough data in this window
                if domain == 'time':
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
                else:  # frequency
                    empty_metrics = {
                        'lf_power': np.nan,
                        'hf_power': np.nan,
                        'lf_hf_ratio': np.nan,
                        'lf_norm': np.nan,
                        'hf_norm': np.nan,
                        'ln_hf': np.nan,
                        'lf_peak': np.nan,
                        'hf_peak': np.nan,
                        'n_samples': 0,
                        'window_start': start_time,
                        'window_end': start_time + window_duration,
                        'window_number': i + 1,
                        'nn_count': len(window_nn)
                    }
                results.append(empty_metrics)
    
    df_results = pd.DataFrame(results)
    
    print(f"Final output: {len(df_results)} windows (expected: 11 for last 40 seconds)")
    
    return df_results

def process_single_file(file_path, show_plots=True):
    """Process a single PPG file and return both time and frequency domain HRV results"""
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    print("-" * 50)
    
    # Parse filename for metadata
    metadata = parse_filename(os.path.basename(file_path))
    if not metadata:
        print("Warning: Could not parse filename for metadata")
        metadata = {'subject': 'unknown', 'condition': 'unknown', 'label': 'unknown'}
    
    print(f"Subject: {metadata['subject']}, Condition: {metadata['condition']}, Label: {metadata['label']}")
    
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
                return None, None, file_path

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
            return None, None, file_path

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
                    print("Performing sliding window HRV analysis on LAST 40 SECONDS (11 windows each domain)...")
                    
                    # Perform sliding window analysis for TIME domain - LAST 40 SECONDS ONLY
                    time_results = sliding_window_hrv_analysis_last40_seconds(
                        nn_intervals, nn_times, 
                        domain='time'
                    )
                    
                    # Perform sliding window analysis for FREQUENCY domain - LAST 40 SECONDS ONLY  
                    freq_results = sliding_window_hrv_analysis_last40_seconds(
                        nn_intervals, nn_times,
                        domain='frequency'
                    )
                    
                    if not time_results.empty and not freq_results.empty:
                        print(f"Successfully analyzed {len(time_results)} time windows and {len(freq_results)} frequency windows")
                        
                        # Create visualization if requested
                        if show_plots:
                            create_time_domain_plots(time_results, time, ppg_filtered, peaks, nn_intervals, nn_times, metadata)
                            create_frequency_domain_plots(freq_results, time, ppg_filtered, peaks, nn_intervals, nn_times, metadata)
                        
                        return time_results, freq_results, file_path
                    else:
                        print("No valid windows found for analysis")
                        return None, None, file_path
                else:
                    print("No valid NN intervals after denoising")
                    return None, None, file_path
            else:
                print("No NN intervals detected")
                return None, None, file_path
        else:
            print("Insufficient peaks detected")
            return None, None, file_path

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, file_path

def create_time_domain_plots(time_results, time, ppg_filtered, peaks, nn_intervals, nn_times, metadata):
    """Create comprehensive plots for time domain analysis in separate window"""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Time Domain HRV Analysis - {metadata["subject"]} {metadata["condition"]} {metadata["label"]}', fontsize=16, fontweight='bold')
    
    # Plot 1: PPG Signal with peaks
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(time, ppg_filtered, label='Filtered PPG', color='blue', alpha=0.7)
    if len(peaks) > 0:
        plt.plot(time[peaks], ppg_filtered[peaks], 'ro', markersize=3, label=f'Peaks ({len(peaks)})')
    plt.title('PPG Signal with Detected Peaks')
    plt.ylabel('Normalized PPG')
    plt.xlabel('Time (s)')
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
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: SDNN over windows
    ax3 = plt.subplot(4, 2, 3)
    valid_sdnn = time_results.dropna(subset=['sdnn'])
    if not valid_sdnn.empty:
        plt.plot(valid_sdnn['window_start'], valid_sdnn['sdnn'], 'b-o', markersize=3, linewidth=1.5)
        plt.title('SDNN Over Time Windows')
        plt.ylabel('SDNN (ms)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: RMSSD over windows
    ax4 = plt.subplot(4, 2, 4)
    valid_rmssd = time_results.dropna(subset=['rmssd'])
    if not valid_rmssd.empty:
        plt.plot(valid_rmssd['window_start'], valid_rmssd['rmssd'], 'r-o', markersize=3, linewidth=1.5)
        plt.title('RMSSD Over Time Windows')
        plt.ylabel('RMSSD (ms)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: pNN20 and pNN50 over windows
    ax5 = plt.subplot(4, 2, 5)
    valid_pnn20 = time_results.dropna(subset=['pnn20'])
    valid_pnn50 = time_results.dropna(subset=['pnn50'])
    if not valid_pnn20.empty:
        plt.plot(valid_pnn20['window_start'], valid_pnn20['pnn20'], 'orange', marker='o', markersize=2, linewidth=1.5, label='pNN20')
    if not valid_pnn50.empty:
        plt.plot(valid_pnn50['window_start'], valid_pnn50['pnn50'], 'purple', marker='s', markersize=2, linewidth=1.5, label='pNN50')
    plt.title('pNN20 and pNN50 Over Time Windows')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Mean Heart Rate over windows
    ax6 = plt.subplot(4, 2, 6)
    valid_hr = time_results.dropna(subset=['nn_mean'])
    if not valid_hr.empty:
        # Convert mean NN interval to heart rate (BPM)
        heart_rate = 60000 / valid_hr['nn_mean']  # 60000 ms per minute
        plt.plot(valid_hr['window_start'], heart_rate, 'm-o', markersize=3, linewidth=1.5)
        plt.title('Heart Rate Over Time Windows')
        plt.ylabel('Heart Rate (BPM)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 7: SDSD over windows
    ax7 = plt.subplot(4, 2, 7)
    valid_sdsd = time_results.dropna(subset=['sdsd'])
    if not valid_sdsd.empty:
        plt.plot(valid_sdsd['window_start'], valid_sdsd['sdsd'], 'cyan', marker='d', markersize=3, linewidth=1.5)
        plt.title('SDSD Over Time Windows')
        plt.ylabel('SDSD (ms)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 8: Triangular Index over windows
    ax8 = plt.subplot(4, 2, 8)
    valid_tri = time_results.dropna(subset=['triangular_index'])
    if not valid_tri.empty:
        plt.plot(valid_tri['window_start'], valid_tri['triangular_index'], 'brown', marker='^', markersize=3, linewidth=1.5)
        plt.title('Triangular Index Over Time Windows')
        plt.ylabel('Triangular Index')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_frequency_domain_plots(freq_results, time, ppg_filtered, peaks, nn_intervals, nn_times, metadata):
    """Create comprehensive plots for frequency domain analysis in separate window"""
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Frequency Domain HRV Analysis - {metadata["subject"]} {metadata["condition"]} {metadata["label"]}', fontsize=16, fontweight='bold')
    
    # Plot 1: PPG Signal with peaks (same as time domain for reference)
    ax1 = plt.subplot(4, 2, 1)
    plt.plot(time, ppg_filtered, label='Filtered PPG', color='blue', alpha=0.7)
    if len(peaks) > 0:
        plt.plot(time[peaks], ppg_filtered[peaks], 'ro', markersize=3, label=f'Peaks ({len(peaks)})')
    plt.title('PPG Signal with Detected Peaks')
    plt.ylabel('Normalized PPG')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: NN Intervals (same as time domain for reference)
    ax2 = plt.subplot(4, 2, 2)
    if len(nn_intervals) > 0:
        plt.plot(nn_times, nn_intervals, 'g-o', markersize=2, linewidth=1, label='NN Intervals')
        plt.axhline(y=np.mean(nn_intervals), color='red', linestyle='--', 
                   label=f"Mean: {np.mean(nn_intervals):.1f} ms")
        plt.ylim(300, 2000)
    plt.title('NN Intervals Over Time')
    plt.ylabel('NN Interval (ms)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: LF Power over windows
    ax3 = plt.subplot(4, 2, 3)
    valid_lf = freq_results.dropna(subset=['lf_power'])
    if not valid_lf.empty:
        plt.plot(valid_lf['window_start'], valid_lf['lf_power'], 'b-o', markersize=3, linewidth=1.5)
        plt.title('LF Power Over Time Windows')
        plt.ylabel('LF Power (ms²)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: HF Power over windows
    ax4 = plt.subplot(4, 2, 4)
    valid_hf = freq_results.dropna(subset=['hf_power'])
    if not valid_hf.empty:
        plt.plot(valid_hf['window_start'], valid_hf['hf_power'], 'r-o', markersize=3, linewidth=1.5)
        plt.title('HF Power Over Time Windows')
        plt.ylabel('HF Power (ms²)')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: LF/HF Ratio over windows
    ax5 = plt.subplot(4, 2, 5)
    valid_ratio = freq_results.dropna(subset=['lf_hf_ratio'])
    if not valid_ratio.empty:
        plt.plot(valid_ratio['window_start'], valid_ratio['lf_hf_ratio'], 'purple', marker='o', markersize=3, linewidth=1.5)
        plt.title('LF/HF Ratio Over Time Windows')
        plt.ylabel('LF/HF Ratio')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Normalized LF and HF Powers
    ax6 = plt.subplot(4, 2, 6)
    valid_lf_norm = freq_results.dropna(subset=['lf_norm'])
    valid_hf_norm = freq_results.dropna(subset=['hf_norm'])
    if not valid_lf_norm.empty:
        plt.plot(valid_lf_norm['window_start'], valid_lf_norm['lf_norm'], 'b-o', markersize=2, linewidth=1.5, label='LF norm')
    if not valid_hf_norm.empty:
        plt.plot(valid_hf_norm['window_start'], valid_hf_norm['hf_norm'], 'r-o', markersize=2, linewidth=1.5, label='HF norm')
    plt.title('Normalized LF and HF Powers')
    plt.ylabel('Normalized Power (%)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Natural Log of HF Power
    ax7 = plt.subplot(4, 2, 7)
    valid_ln_hf = freq_results.dropna(subset=['ln_hf'])
    if not valid_ln_hf.empty:
        plt.plot(valid_ln_hf['window_start'], valid_ln_hf['ln_hf'], 'g-o', markersize=3, linewidth=1.5)
        plt.title('Natural Log of HF Power (LnHF)')
        plt.ylabel('ln(HF) [ln(ms²)]')
        plt.xlabel('Time (s)')
        plt.grid(True, alpha=0.3)
    
    # Plot 8: LF vs HF Power Scatter Plot
    ax8 = plt.subplot(4, 2, 8)
    valid_both = freq_results.dropna(subset=['lf_power', 'hf_power'])
    if not valid_both.empty:
        plt.scatter(valid_both['lf_power'], valid_both['hf_power'], alpha=0.6, c='purple')
        plt.title('LF vs HF Power Relationship')
        plt.xlabel('LF Power (ms²)')
        plt.ylabel('HF Power (ms²)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def display_summary_statistics(time_results, freq_results):
    """Display summary statistics for both time and frequency domain results"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS ACROSS ALL WINDOWS")
    print("="*80)
    
    print("\nTIME DOMAIN METRICS:")
    print("-" * 40)
    time_metrics = ['sdnn', 'sdsd', 'rmssd', 'pnn20', 'pnn50', 'nn_mean']
    for metric in time_metrics:
        if metric in time_results.columns:
            valid_values = time_results[metric].dropna()
            if len(valid_values) > 0:
                unit = "ms" if metric in ['sdnn', 'sdsd', 'rmssd', 'nn_mean'] else "%"
                print(f"{metric.upper()} ({unit}):")
                print(f"  Mean: {valid_values.mean():.2f}")
                print(f"  Std:  {valid_values.std():.2f}")
                print(f"  Min:  {valid_values.min():.2f}")
                print(f"  Max:  {valid_values.max():.2f}")
                print()
    
    print("\nFREQUENCY DOMAIN METRICS:")
    print("-" * 40)
    freq_metrics = ['lf_power', 'hf_power', 'lf_hf_ratio', 'lf_norm', 'hf_norm', 'ln_hf']
    for metric in freq_metrics:
        if metric in freq_results.columns:
            valid_values = freq_results[metric].dropna()
            if len(valid_values) > 0:
                if metric in ['lf_power', 'hf_power']:
                    unit = "ms²"
                elif metric in ['lf_norm', 'hf_norm']:
                    unit = "%"
                elif metric == 'ln_hf':
                    unit = "ln(ms²)"
                else:
                    unit = ""
                    
                print(f"{metric.upper().replace('_', ' ')} ({unit}):")
                print(f"  Mean: {valid_values.mean():.4f}")
                print(f"  Std:  {valid_values.std():.4f}")
                print(f"  Min:  {valid_values.min():.4f}")
                print(f"  Max:  {valid_values.max():.4f}")
                print()

def check_consistency_across_files(output_dir):
    """Check that all files have the same number of windows per subject-condition (should be 11)"""
    print("\n" + "="*80)
    print("CHECKING CONSISTENCY ACROSS FILES")
    print("="*80)
    
    # Get all CSV files
    time_files = glob.glob(os.path.join(output_dir, "windowed", "*_windowed.csv"))
    freq_files = glob.glob(os.path.join(output_dir, "frequency_windowed", "*_frequency_windowed.csv"))
    
    file_info = []
    
    # Check time domain files
    for file_path in time_files:
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            metadata = parse_filename(filename.replace('_windowed.csv', '.txt'))
            if metadata:
                file_info.append({
                    'subject': metadata['subject'],
                    'condition': metadata['condition'],
                    'label': metadata['label'],
                    'domain': 'time',
                    'n_windows': len(df),
                    'filename': filename
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Check frequency domain files
    for file_path in freq_files:
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            metadata = parse_filename(filename.replace('_frequency_windowed.csv', '.txt'))
            if metadata:
                file_info.append({
                    'subject': metadata['subject'],
                    'condition': metadata['condition'],
                    'label': metadata['label'],
                    'domain': 'frequency',
                    'n_windows': len(df),
                    'filename': filename
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not file_info:
        print("No CSV files found to check.")
        return
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(file_info)
    
    # Group by subject-condition to check consistency
    print("Windows per subject-condition combination (expected: 11 windows each):")
    print("-" * 70)
    
    for (subject, condition), group in summary_df.groupby(['subject', 'condition']):
        print(f"\n{subject} - {condition}:")
        for _, row in group.iterrows():
            status = "✓" if row['n_windows'] == 11 else "✗"
            print(f"  {status} {row['label']:5} - {row['domain']:9} - {row['n_windows']:2} windows - {row['filename']}")
        
        # Check if all have same number of windows (should be 11)
        window_counts = group['n_windows'].unique()
        if len(window_counts) == 1 and window_counts[0] == 11:
            print(f"  ✓ Perfect: {window_counts[0]} windows (last 40 seconds)")
        elif len(window_counts) == 1:
            print(f"  ⚠ Consistent but unexpected: {window_counts[0]} windows (expected 11)")
        else:
            print(f"  ✗ Inconsistent: {window_counts} windows (expected 11)")
    
    # Overall summary
    print(f"\n" + "-" * 70)
    total_expected = len(summary_df) * 11  # 11 windows per file
    total_actual = summary_df['n_windows'].sum()
    perfect_files = (summary_df['n_windows'] == 11).sum()
    
    print(f"Total files processed: {len(summary_df)}")
    print(f"Files with exactly 11 windows: {perfect_files}/{len(summary_df)}")
    print(f"Expected total windows: {total_expected} (11 per file)")
    print(f"Actual total windows: {total_actual}")
    print(f"Consistency: {'✓ PASS' if total_expected == total_actual else '✗ FAIL'}")
    print(f"Note: 11 windows = last 40 seconds with 30s windows moving 1s each time")

def main():
    print("PPG Signal Processing - HRV Analysis (Last 40 Seconds Only)")
    print("30-second windows moving 1 second forward = 11 windows per file")
    print("Generates both time-domain and frequency-domain features")
    print("=" * 80)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    print(f"Scanning directory: {current_dir}")

    # Look for .txt files with the expected naming pattern
    txt_files = glob.glob(os.path.join(current_dir, "pr_*_*_*-sequence.txt"))
    if not txt_files:
        # Fallback to any .txt files
        txt_files = glob.glob(os.path.join(current_dir, "**/*.txt"), recursive=True)
        txt_files.extend(glob.glob(os.path.join(current_dir, "*.txt")))
        txt_files = sorted(list(set(txt_files)))

    if not txt_files:
        print("No .txt files found.")
        return

    print(f"Found {len(txt_files)} .txt file(s):")
    for i, file in enumerate(txt_files):
        file_size = os.path.getsize(file) / 1024
        metadata = parse_filename(os.path.basename(file))
        if metadata:
            print(f"  {i}: {os.path.relpath(file, current_dir)} ({file_size:.1f} KB) - {metadata['subject']} {metadata['condition']} {metadata['label']}")
        else:
            print(f"  {i}: {os.path.relpath(file, current_dir)} ({file_size:.1f} KB) - [unparseable]")

    # Enhanced file selection
    print("\nProcessing Options:")
    print("0: Select and process a single file")
    print("1: Process all files")
    print("2: Check consistency of existing processed files")

    try:
        choice = int(input("Enter your choice (0, 1, or 2): "))
        if choice not in [0, 1, 2]:
            raise ValueError("Choice must be 0, 1, or 2")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create output directories
    output_dir = os.path.join(current_dir, "datasets")
    time_output_dir = os.path.join(output_dir, "windowed")
    freq_output_dir = os.path.join(output_dir, "frequency_windowed")
    os.makedirs(time_output_dir, exist_ok=True)
    os.makedirs(freq_output_dir, exist_ok=True)
    
    print(f"Time domain output directory: {time_output_dir}")
    print(f"Frequency domain output directory: {freq_output_dir}")

    if choice == 2:
        # Check consistency only
        check_consistency_across_files(output_dir)
        return

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
        time_results, freq_results, processed_file = process_single_file(file_path, show_plots=True)
        
        if time_results is not None and freq_results is not None:
            # Save results
            base_name = os.path.splitext(os.path.basename(processed_file))[0]
            
            # Save time domain results
            time_output_filename = os.path.join(time_output_dir, f"{base_name}_windowed.csv")
            time_results.to_csv(time_output_filename, index=False)
            print(f"\nTime domain results saved to: {time_output_filename}")
            print(f"Time domain output contains exactly {len(time_results)} samples (last 40 seconds)")
            
            # Save frequency domain results
            freq_output_filename = os.path.join(freq_output_dir, f"{base_name}_frequency_windowed.csv")
            freq_results.to_csv(freq_output_filename, index=False)
            print(f"Frequency domain results saved to: {freq_output_filename}")
            print(f"Frequency domain output contains exactly {len(freq_results)} samples (last 40 seconds)")
            
            # Display summary statistics
            display_summary_statistics(time_results, freq_results)
        else:
            print("Failed to process the selected file.")

    else:
        # Batch processing all files
        print(f"\nProcessing all {len(txt_files)} files...")
        successful_files = 0
        failed_files = 0
        
        for i, file_path in enumerate(txt_files):
            print(f"\n[{i+1}/{len(txt_files)}] Processing: {os.path.basename(file_path)}")
            
            time_results, freq_results, processed_file = process_single_file(file_path, show_plots=False)
            
            if time_results is not None and freq_results is not None:
                # Save results
                base_name = os.path.splitext(os.path.basename(processed_file))[0]
                
                # Save time domain results
                time_output_filename = os.path.join(time_output_dir, f"{base_name}_windowed.csv")
                time_results.to_csv(time_output_filename, index=False)
                
                # Save frequency domain results
                freq_output_filename = os.path.join(freq_output_dir, f"{base_name}_frequency_windowed.csv")
                freq_results.to_csv(freq_output_filename, index=False)
                
                print(f"✓ Results saved:")
                print(f"  Time: {os.path.relpath(time_output_filename, current_dir)} ({len(time_results)} samples)")
                print(f"  Freq: {os.path.relpath(freq_output_filename, current_dir)} ({len(freq_results)} samples)")
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
        print(f"Time domain output directory: {os.path.relpath(time_output_dir, current_dir)}")
        print(f"Frequency domain output directory: {os.path.relpath(freq_output_dir, current_dir)}")
        print(f"Each successful file contains exactly 11 samples in each domain (last 40 seconds)")
        
        # Run consistency check
        if successful_files > 0:
            check_consistency_across_files(output_dir)

if __name__ == "__main__":
    main()