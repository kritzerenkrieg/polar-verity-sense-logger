import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, stft, cwt, morlet
from scipy.interpolate import interp1d
from scipy import stats
import pywt
import os
import glob
import sys
from collections import deque

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
    
    if mad == 0:
        mad = np.std(nn_intervals) * 0.6745
    
    modified_z_scores = 0.6745 * (nn_intervals - median_nn) / mad
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
        valid_indices = ~artifact_mask
        if np.sum(valid_indices) < 2:
            print("Warning: Too many artifacts for interpolation.")
            return nn_intervals, nn_times
        
        valid_times = nn_times[valid_indices]
        valid_nn = nn_intervals[valid_indices]
        
        if len(valid_times) >= 2:
            interp_func = interp1d(valid_times, valid_nn, kind='linear', 
                                 bounds_error=False, fill_value='extrapolate')
            corrected_nn[artifact_mask] = interp_func(nn_times[artifact_mask])
    
    elif method == 'remove':
        valid_mask = ~artifact_mask
        corrected_nn = nn_intervals[valid_mask]
        nn_times = nn_times[valid_mask]
    
    return corrected_nn, nn_times

def denoise_nn_intervals(nn_intervals, nn_times, mad_threshold=3.0, 
                        change_threshold=0.2, correction_method='interpolation'):
    """Comprehensive denoising of NN intervals"""
    if len(nn_intervals) < 3:
        return nn_intervals, nn_times, np.array([])
    
    artifacts_mad = detect_artifacts_mad(nn_intervals, mad_threshold)
    artifacts_change = detect_artifacts_percentage_change(nn_intervals, change_threshold)
    combined_artifacts = artifacts_mad | artifacts_change
    
    artifact_count = np.sum(combined_artifacts)
    artifact_percentage = (artifact_count / len(nn_intervals)) * 100
    
    if artifact_percentage > 50:
        correction_method = 'remove'
    
    corrected_nn, corrected_times = correct_artifacts(nn_intervals, nn_times, 
                                                     combined_artifacts, correction_method)
    
    return corrected_nn, corrected_times, combined_artifacts

def calculate_nn_intervals(peaks, time_array):
    if len(peaks) < 2:
        return np.array([]), np.array([])
    peak_times = time_array[peaks]
    nn_intervals = np.diff(peak_times) * 1000  # in ms
    
    valid_mask = (nn_intervals >= 300) & (nn_intervals <= 2000)
    nn_intervals_filtered = nn_intervals[valid_mask]
    peak_times_filtered = peak_times[1:][valid_mask]
    
    return nn_intervals_filtered, peak_times_filtered

def calculate_frequency_domain_features(nn_intervals, nn_times, method='stft'):
    """Calculate frequency domain features using STFT or Wavelet Transform"""
    if len(nn_intervals) < 10:
        return {'lf_power': np.nan, 'hf_power': np.nan, 'lf_hf_ratio': np.nan}
    
    try:
        # Interpolate to get evenly spaced signal for frequency analysis
        fs = 4.0  # 4 Hz resampling rate (common for HRV analysis)
        
        # Create time vector for interpolation
        time_min, time_max = np.min(nn_times), np.max(nn_times)
        time_interp = np.arange(time_min, time_max, 1/fs)
        
        if len(time_interp) < 10:
            return {'lf_power': np.nan, 'hf_power': np.nan, 'lf_hf_ratio': np.nan}
        
        # Interpolate NN intervals
        interp_func = interp1d(nn_times, nn_intervals, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        nn_interp = interp_func(time_interp)
        
        # Remove mean
        nn_interp = nn_interp - np.mean(nn_interp)
        
        if method == 'stft':
            # Short-Term Fourier Transform
            f, t, Zxx = stft(nn_interp, fs=fs, nperseg=min(64, len(nn_interp)//2))
            psd = np.mean(np.abs(Zxx)**2, axis=1)
        
        elif method == 'wavelet':
            # Wavelet Transform using Morlet wavelet
            scales = np.arange(1, min(32, len(nn_interp)//4))
            coefficients, frequencies = pywt.cwt(nn_interp, scales, 'morl', sampling_period=1/fs)
            psd = np.mean(np.abs(coefficients)**2, axis=1)
            f = frequencies
        
        else:  # fallback to simple periodogram
            f = np.fft.fftfreq(len(nn_interp), 1/fs)
            fft = np.fft.fft(nn_interp)
            psd = np.abs(fft)**2
            
            # Keep only positive frequencies
            pos_mask = f >= 0
            f = f[pos_mask]
            psd = psd[pos_mask]
        
        # Define frequency bands
        lf_band = (f >= 0.04) & (f <= 0.15)  # Low frequency
        hf_band = (f >= 0.15) & (f <= 0.4)   # High frequency
        
        # Calculate power in each band
        lf_power = np.sum(psd[lf_band]) if np.any(lf_band) else 0
        hf_power = np.sum(psd[hf_band]) if np.any(hf_band) else 0
        
        # Calculate LF/HF ratio
        lf_hf_ratio = lf_power / (hf_power + 1e-8)
        
        return {
            'lf_power': lf_power,
            'hf_power': hf_power, 
            'lf_hf_ratio': lf_hf_ratio
        }
        
    except Exception as e:
        print(f"Error in frequency domain analysis: {e}")
        return {'lf_power': np.nan, 'hf_power': np.nan, 'lf_hf_ratio': np.nan}

def calculate_poincare_features(nn_intervals):
    """Calculate Poincaré plot features: SD1, SD2, SD1/SD2"""
    if len(nn_intervals) < 3:
        return {'sd1': np.nan, 'sd2': np.nan, 'sd1_sd2_ratio': np.nan}
    
    try:
        nn = np.array(nn_intervals)
        
        # Create Poincaré plot points
        x = nn[:-1]  # RR(n)
        y = nn[1:]   # RR(n+1)
        
        if len(x) < 2:
            return {'sd1': np.nan, 'sd2': np.nan, 'sd1_sd2_ratio': np.nan}
        
        # Calculate differences
        diff_x_y = x - y
        sum_x_y = x + y
        
        # SD1: Standard deviation perpendicular to line of identity
        # SD1 represents short-term variability
        sd1 = np.sqrt(np.var(diff_x_y) / 2.0)
        
        # SD2: Standard deviation along the line of identity  
        # SD2 represents long-term variability
        sd2 = np.sqrt(2 * np.var(sum_x_y) / 4.0 - np.var(diff_x_y) / 4.0)
        
        # Ensure SD2 is positive
        if sd2 < 0:
            sd2 = np.sqrt(np.var(sum_x_y) / 2.0)
        
        # SD1/SD2 ratio
        sd1_sd2_ratio = sd1 / (sd2 + 1e-8)
        
        return {
            'sd1': sd1,
            'sd2': sd2,
            'sd1_sd2_ratio': sd1_sd2_ratio
        }
        
    except Exception as e:
        print(f"Error in Poincaré analysis: {e}")
        return {'sd1': np.nan, 'sd2': np.nan, 'sd1_sd2_ratio': np.nan}

def calculate_approximate_entropy(nn_intervals, m=2, r=None):
    """Calculate Approximate Entropy (ApEn)"""
    if len(nn_intervals) < 10:
        return np.nan
        
    try:
        data = np.array(nn_intervals)
        N = len(data)
        
        if r is None:
            r = 0.2 * np.std(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                matches = 0
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j], m) <= r:
                        matches += 1
                C[i] = matches / float(N - m + 1)
            
            phi = np.mean(np.log(C + 1e-8))
            return phi
        
        return _phi(m) - _phi(m + 1)
        
    except Exception as e:
        print(f"Error in ApEn calculation: {e}")
        return np.nan

def calculate_sample_entropy(nn_intervals, m=2, r=None):
    """Calculate Sample Entropy (SampEn)"""
    if len(nn_intervals) < 10:
        return np.nan
        
    try:
        data = np.array(nn_intervals)
        N = len(data)
        
        if r is None:
            r = 0.2 * np.std(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            matches = 0
            total_pairs = 0
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if i != j:  # Exclude self-matches (key difference from ApEn)
                        total_pairs += 1
                        if _maxdist(template_i, patterns[j], m) <= r:
                            matches += 1
            
            return matches / float(total_pairs) if total_pairs > 0 else 0
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return np.nan
            
        return -np.log(phi_m1 / phi_m)
        
    except Exception as e:
        print(f"Error in SampEn calculation: {e}")
        return np.nan

def calculate_multiscale_entropy(nn_intervals, max_scale=5, m=2, r=None):
    """Calculate Multiscale Entropy (MSE)"""
    if len(nn_intervals) < 20:
        return np.nan
        
    try:
        data = np.array(nn_intervals)
        
        if r is None:
            r = 0.2 * np.std(data)
        
        entropies = []
        
        for scale in range(1, min(max_scale + 1, len(data) // 10)):
            # Coarse-grain the series
            if scale == 1:
                coarse_grained = data
            else:
                n_points = len(data) // scale
                coarse_grained = np.array([np.mean(data[i*scale:(i+1)*scale]) 
                                         for i in range(n_points)])
            
            if len(coarse_grained) >= 10:
                entropy = calculate_sample_entropy(coarse_grained, m=m, r=r)
                if not np.isnan(entropy):
                    entropies.append(entropy)
        
        # Return mean of entropies across scales
        return np.mean(entropies) if entropies else np.nan
        
    except Exception as e:
        print(f"Error in MSE calculation: {e}")
        return np.nan

def calculate_dfa(nn_intervals, min_window=4, max_window=None):
    """Calculate Detrended Fluctuation Analysis (DFA)"""
    if len(nn_intervals) < 10:
        return np.nan
        
    try:
        data = np.array(nn_intervals)
        N = len(data)
        
        if max_window is None:
            max_window = N // 4
        
        # Step 1: Integration (cumulative sum after removing mean)
        y = np.cumsum(data - np.mean(data))
        
        # Step 2: Divide into windows and calculate fluctuation
        windows = np.logspace(np.log10(min_window), np.log10(max_window), 
                             num=min(15, max_window - min_window + 1))
        windows = np.unique(windows.astype(int))
        
        fluctuations = []
        
        for window in windows:
            if window >= N:
                continue
                
            # Number of complete windows
            n_windows = N // window
            
            # Calculate fluctuation for each window
            F_n = 0
            for i in range(n_windows):
                start_idx = i * window
                end_idx = (i + 1) * window
                
                # Linear detrending
                x = np.arange(window)
                coeffs = np.polyfit(x, y[start_idx:end_idx], 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                F_n += np.sum((y[start_idx:end_idx] - trend) ** 2)
            
            # Root mean square fluctuation
            F_n = np.sqrt(F_n / (n_windows * window))
            fluctuations.append(F_n)
        
        if len(fluctuations) < 3:
            return np.nan
        
        # Step 3: Calculate scaling exponent (alpha)
        log_windows = np.log10(windows[:len(fluctuations)])
        log_fluctuations = np.log10(fluctuations)
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(log_windows) & np.isfinite(log_fluctuations)
        if np.sum(valid_mask) < 3:
            return np.nan
            
        log_windows = log_windows[valid_mask]
        log_fluctuations = log_fluctuations[valid_mask]
        
        # Linear regression to find scaling exponent
        slope, _, r_value, _, _ = stats.linregress(log_windows, log_fluctuations)
        
        return slope
        
    except Exception as e:
        print(f"Error in DFA calculation: {e}")
        return np.nan

def calculate_correlation_dimension(nn_intervals, max_dim=10):
    """Calculate Correlation Dimension (CD)"""
    if len(nn_intervals) < 20:
        return np.nan
        
    try:
        data = np.array(nn_intervals)
        N = len(data)
        
        # Normalize data
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        # Estimate correlation dimension using Grassberger-Procaccia algorithm
        distances = []
        
        # Calculate all pairwise distances
        for i in range(N):
            for j in range(i + 1, N):
                dist = abs(data[i] - data[j])
                distances.append(dist)
        
        distances = np.array(distances)
        
        if len(distances) == 0:
            return np.nan
        
        # Use range of r values
        r_min = np.percentile(distances, 1)
        r_max = np.percentile(distances, 50)  # Don't go too high
        
        if r_min >= r_max or r_min <= 0:
            return np.nan
        
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), 10)
        
        correlations = []
        
        for r in r_values:
            # Count pairs within distance r
            count = np.sum(distances <= r)
            correlation = count / len(distances)
            correlations.append(correlation + 1e-8)  # Avoid log(0)
        
        # Estimate correlation dimension from slope
        log_r = np.log(r_values)
        log_c = np.log(correlations)
        
        # Find linear region (middle portion)
        start_idx = len(log_r) // 4
        end_idx = 3 * len(log_r) // 4
        
        if end_idx <= start_idx + 1:
            return np.nan
        
        slope, _, r_value, _, _ = stats.linregress(log_r[start_idx:end_idx], 
                                                 log_c[start_idx:end_idx])
        
        return slope
        
    except Exception as e:
        print(f"Error in CD calculation: {e}")
        return np.nan

def calculate_validated_hrv_features(nn_intervals, nn_times):
    """Calculate the 12 validated HRV features"""
    if len(nn_intervals) < 5:
        return create_empty_validated_features()
    
    features = {}
    
    # Time-Frequency Domain Features
    print("  Calculating frequency domain features...")
    
    # STFT-based features
    stft_features = calculate_frequency_domain_features(nn_intervals, nn_times, method='stft')
    features['stft_lf_power'] = stft_features['lf_power']
    features['stft_hf_power'] = stft_features['hf_power'] 
    features['stft_lf_hf_ratio'] = stft_features['lf_hf_ratio']
    
    # Wavelet Transform features
    wt_features = calculate_frequency_domain_features(nn_intervals, nn_times, method='wavelet')
    features['wt_lf_power'] = wt_features['lf_power']
    features['wt_hf_power'] = wt_features['hf_power']
    features['wt_lf_hf_ratio'] = wt_features['lf_hf_ratio']
    
    # Non-linear Domain Features
    print("  Calculating Poincaré features...")
    poincare_features = calculate_poincare_features(nn_intervals)
    features['sd1'] = poincare_features['sd1']
    features['sd2'] = poincare_features['sd2'] 
    features['sd1_sd2_ratio'] = poincare_features['sd1_sd2_ratio']
    
    # Entropy Features
    print("  Calculating entropy features...")
    features['apen'] = calculate_approximate_entropy(nn_intervals)
    features['sampen'] = calculate_sample_entropy(nn_intervals)
    features['mse'] = calculate_multiscale_entropy(nn_intervals)
    
    # Fractal Dimension Features
    print("  Calculating fractal features...")
    features['dfa'] = calculate_dfa(nn_intervals)
    features['cd'] = calculate_correlation_dimension(nn_intervals)
    
    return features

def create_empty_validated_features():
    """Create empty validated features dictionary"""
    return {
        # Time-Frequency Domain
        'stft_lf_power': np.nan,
        'stft_hf_power': np.nan,
        'stft_lf_hf_ratio': np.nan,
        'wt_lf_power': np.nan,
        'wt_hf_power': np.nan,
        'wt_lf_hf_ratio': np.nan,
        # Non-linear Domain
        'sd1': np.nan,
        'sd2': np.nan,
        'sd1_sd2_ratio': np.nan,
        'apen': np.nan,
        'sampen': np.nan,
        'mse': np.nan,
        'dfa': np.nan,
        'cd': np.nan
    }

def extract_window_nn_intervals(nn_intervals, nn_times, start_time, window_duration=30):
    """Extract NN intervals within a specific time window"""
    end_time = start_time + window_duration
    
    nn_intervals = np.array(nn_intervals)
    nn_times = np.array(nn_times)
    
    window_mask = (nn_times >= start_time) & (nn_times < end_time)
    
    window_nn = nn_intervals[window_mask]
    window_times = nn_times[window_mask]
    
    return window_nn, window_times

def sliding_window_validated_hrv_analysis(nn_intervals, nn_times, target_samples=70):
    """
    Perform sliding window analysis with validated HRV features - EXACTLY 70 SAMPLES
    
    Parameters:
    - nn_intervals: array of NN intervals in ms
    - nn_times: array of NN interval timestamps in seconds
    - target_samples: exact number of samples to output (default: 70)
    
    Returns:
    - DataFrame with validated HRV metrics for exactly 70 windows
    """
    
    if len(nn_intervals) == 0 or len(nn_times) == 0:
        return pd.DataFrame()
    
    nn_intervals = np.array(nn_intervals)
    nn_times = np.array(nn_times)
    
    total_duration = nn_times[-1] - nn_times[0]
    window_duration = 30  # Fixed window duration in seconds
    
    if target_samples <= 1:
        step_size = total_duration
    else:
        step_size = (total_duration - window_duration) / (target_samples - 1)
    
    if step_size <= 0:
        step_size = 1.0
        if total_duration < window_duration:
            window_duration = total_duration * 0.8
    
    window_starts = np.linspace(0, total_duration - window_duration, target_samples)
    
    print(f"Total recording duration: {total_duration:.1f} seconds")
    print(f"Window duration: {window_duration:.1f} seconds")
    print(f"Step size: {step_size:.3f} seconds")
    print(f"Target windows: {target_samples}")
    print(f"Calculated windows: {len(window_starts)}")
    
    results = []
    
    for i, start_time in enumerate(window_starts):
        print(f"Processing window {i+1}/{target_samples}...")
        
        # Extract NN intervals for this window
        window_nn, window_times = extract_window_nn_intervals(
            nn_intervals, nn_times, start_time, window_duration
        )
        
        # Calculate validated HRV features for this window
        if len(window_nn) >= 5:  # Need at least 5 NN intervals for meaningful analysis
            hrv_features = calculate_validated_hrv_features(window_nn, window_times)
            
            # Add window information
            hrv_features['window_start'] = start_time
            hrv_features['window_end'] = start_time + window_duration
            hrv_features['window_number'] = i + 1
            hrv_features['nn_count'] = len(window_nn)
            
            results.append(hrv_features)
        else:
            # Not enough data in this window
            empty_features = create_empty_validated_features()
            empty_features.update({
                'window_start': start_time,
                'window_end': start_time + window_duration,
                'window_number': i + 1,
                'nn_count': len(window_nn)
            })
            results.append(empty_features)
    
    df_results = pd.DataFrame(results)
    
    # Ensure we have exactly target_samples rows
    if len(df_results) > target_samples:
        df_results = df_results.head(target_samples)
    elif len(df_results) < target_samples:
        for i in range(len(df_results), target_samples):
            empty_row = create_empty_validated_features()
            empty_row.update({
                'window_start': np.nan,
                'window_end': np.nan,
                'window_number': i + 1,
                'nn_count': 0
            })
            df_results = pd.concat([df_results, pd.DataFrame([empty_row])], ignore_index=True)
    
    print(f"Final output: {len(df_results)} windows (target: {target_samples})")
    print(f"Validated features: 14 research-proven HRV features")
    
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
    """Process a single PPG file and return validated HRV results - EXACTLY 70 SAMPLES"""
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
                    print("Performing validated sliding window HRV analysis (70 samples)...")
                    
                    # Perform validated sliding window analysis - EXACTLY 70 SAMPLES
                    hrv_results = sliding_window_validated_hrv_analysis(
                        nn_intervals, nn_times, 
                        target_samples=70  # HARD-CODED to 70 samples
                    )
                    
                    if not hrv_results.empty:
                        print(f"Successfully analyzed {len(hrv_results)} windows with validated features")
                        
                        # Create visualization if requested
                        if show_plots:
                            create_validated_sliding_window_plots(hrv_results, time, ppg_filtered, peaks, nn_intervals, nn_times)
                        
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
    print("Validated HRV Signal Processing - 14 Research-Proven Features (70 Samples)")
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

    # Create output directory for validated features
    output_dir = os.path.join(current_dir, "datasets", "validated_windowed")
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
            # Save results with validated suffix
            base_name = os.path.splitext(os.path.basename(processed_file))[0]
            output_filename = os.path.join(output_dir, f"{base_name}_validated_windowed.csv")
            hrv_results.to_csv(output_filename, index=False)
            print(f"\nValidated results saved to: {output_filename}")
            print(f"Output contains exactly {len(hrv_results)} samples with {len(hrv_results.columns)} features")
            
            # Display validated summary statistics
            display_validated_summary_statistics(hrv_results)
        else:
            print("Failed to process the selected file.")

    else:
        # Batch processing all files
        print(f"\nProcessing all {len(txt_files)} files with validated features...")
        successful_files = 0
        failed_files = 0
        
        for i, file_path in enumerate(txt_files):
            print(f"\n[{i+1}/{len(txt_files)}] Processing: {os.path.basename(file_path)}")
            
            hrv_results, processed_file = process_single_file(file_path, show_plots=False)
            
            if hrv_results is not None:
                # Save results with validated suffix
                base_name = os.path.splitext(os.path.basename(processed_file))[0]
                output_filename = os.path.join(output_dir, f"{base_name}_validated_windowed.csv")
                hrv_results.to_csv(output_filename, index=False)
                print(f"✓ Validated results saved to: {os.path.relpath(output_filename, current_dir)} ({len(hrv_results)} samples, {len(hrv_results.columns)} features)")
                successful_files += 1
            else:
                print(f"✗ Failed to process: {os.path.basename(processed_file)}")
                failed_files += 1
        
        print(f"\n" + "="*80)
        print("VALIDATED BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Total files: {len(txt_files)}")
        print(f"Successfully processed: {successful_files}")
        print(f"Failed: {failed_files}")
        print(f"Output directory: {os.path.relpath(output_dir, current_dir)}")
        print(f"Each successful file contains exactly 70 samples with 14 validated HRV features")

def display_validated_summary_statistics(hrv_results):
    """Display summary statistics for validated HRV results"""
    print("\nVALIDATED HRV FEATURES SUMMARY STATISTICS:")
    print("="*60)
    
    # Time-Frequency Domain Features
    print("TIME-FREQUENCY DOMAIN FEATURES:")
    print("-" * 40)
    tf_features = ['stft_lf_power', 'stft_hf_power', 'stft_lf_hf_ratio', 
                   'wt_lf_power', 'wt_hf_power', 'wt_lf_hf_ratio']
    for feature in tf_features:
        if feature in hrv_results.columns:
            valid_values = hrv_results[feature].dropna()
            if len(valid_values) > 0:
                method = "STFT" if "stft" in feature else "WT"
                print(f"{method} {feature.split('_', 1)[1].upper()}: "
                      f"Mean={valid_values.mean():.4f}, Std={valid_values.std():.4f}")
    
    # Non-linear Domain Features
    print(f"\nNON-LINEAR DOMAIN FEATURES:")
    print("-" * 40)
    
    # Poincaré features
    poincare_features = ['sd1', 'sd2', 'sd1_sd2_ratio']
    print("Poincaré Plot Features:")
    for feature in poincare_features:
        if feature in hrv_results.columns:
            valid_values = hrv_results[feature].dropna()
            if len(valid_values) > 0:
                unit = "(ms)" if feature in ['sd1', 'sd2'] else "(ratio)"
                print(f"  {feature.upper()} {unit}: "
                      f"Mean={valid_values.mean():.4f}, Std={valid_values.std():.4f}")
    
    # Entropy features
    entropy_features = ['apen', 'sampen', 'mse']
    print("Entropy Features:")
    for feature in entropy_features:
        if feature in hrv_results.columns:
            valid_values = hrv_results[feature].dropna()
            if len(valid_values) > 0:
                full_name = {"apen": "Approximate Entropy", 
                            "sampen": "Sample Entropy", 
                            "mse": "Multiscale Entropy"}[feature]
                print(f"  {full_name}: "
                      f"Mean={valid_values.mean():.4f}, Std={valid_values.std():.4f}")
    
    # Fractal features
    fractal_features = ['dfa', 'cd']
    print("Fractal Dimension Features:")
    for feature in fractal_features:
        if feature in hrv_results.columns:
            valid_values = hrv_results[feature].dropna()
            if len(valid_values) > 0:
                full_name = {"dfa": "Detrended Fluctuation Analysis", 
                            "cd": "Correlation Dimension"}[feature]
                print(f"  {full_name}: "
                      f"Mean={valid_values.mean():.4f}, Std={valid_values.std():.4f}")
    
    # Feature availability summary
    total_features = len(hrv_results.columns)
    validated_features = len([col for col in hrv_results.columns 
                            if col not in ['window_start', 'window_end', 'window_number', 'nn_count']])
    
    print(f"\nFEATURE AVAILABILITY SUMMARY:")
    print("-" * 40)
    print(f"Total columns: {total_features}")
    print(f"Validated HRV features: {validated_features}")
    print(f"Window metadata: {total_features - validated_features}")
    
    # Data quality summary
    print(f"\nDATA QUALITY SUMMARY:")
    print("-" * 40)
    valid_windows = len(hrv_results[hrv_results['nn_count'] >= 5])
    print(f"Windows with ≥5 NN intervals: {valid_windows}/{len(hrv_results)} "
          f"({100*valid_windows/len(hrv_results):.1f}%)")

def create_validated_sliding_window_plots(hrv_results, time, ppg_filtered, peaks, nn_intervals, nn_times):
    """Create comprehensive plots for validated sliding window analysis"""
    
    fig = plt.figure(figsize=(20, 24))
    
    # Plot 1: PPG Signal with peaks
    ax1 = plt.subplot(6, 2, 1)
    plt.plot(time, ppg_filtered, label='Filtered PPG', color='blue', alpha=0.7)
    if len(peaks) > 0:
        plt.plot(time[peaks], ppg_filtered[peaks], 'ro', markersize=3, label=f'Peaks ({len(peaks)})')
    plt.title('PPG Signal with Detected Peaks')
    plt.ylabel('Normalized PPG')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: NN Intervals
    ax2 = plt.subplot(6, 2, 2)
    if len(nn_intervals) > 0:
        plt.plot(nn_times, nn_intervals, 'g-o', markersize=2, linewidth=1, label='NN Intervals')
        plt.axhline(y=np.mean(nn_intervals), color='red', linestyle='--', 
                   label=f"Mean: {np.mean(nn_intervals):.1f} ms")
        plt.ylim(300, 2000)
    plt.title('NN Intervals Over Time')
    plt.ylabel('NN Interval (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: STFT LF Power
    ax3 = plt.subplot(6, 2, 3)
    if 'stft_lf_power' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['stft_lf_power'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['stft_lf_power'], 
                    'b-o', markersize=3, linewidth=1.5)
            plt.title('STFT LF Power Over Time Windows')
            plt.ylabel('LF Power (ms²)')
            plt.grid(True, alpha=0.3)
    
    # Plot 4: STFT HF Power
    ax4 = plt.subplot(6, 2, 4)
    if 'stft_hf_power' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['stft_hf_power'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['stft_hf_power'], 
                    'r-o', markersize=3, linewidth=1.5)
            plt.title('STFT HF Power Over Time Windows')
            plt.ylabel('HF Power (ms²)')
            plt.grid(True, alpha=0.3)
    
    # Plot 5: SD1 (Poincaré)
    ax5 = plt.subplot(6, 2, 5)
    if 'sd1' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['sd1'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['sd1'], 
                    'purple', marker='o', markersize=3, linewidth=1.5)
            plt.title('SD1 (Short-term Variability) Over Time')
            plt.ylabel('SD1 (ms)')
            plt.grid(True, alpha=0.3)
    
    # Plot 6: SD2 (Poincaré)
    ax6 = plt.subplot(6, 2, 6)
    if 'sd2' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['sd2'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['sd2'], 
                    'orange', marker='o', markersize=3, linewidth=1.5)
            plt.title('SD2 (Long-term Variability) Over Time')
            plt.ylabel('SD2 (ms)')
            plt.grid(True, alpha=0.3)
    
    # Plot 7: Sample Entropy
    ax7 = plt.subplot(6, 2, 7)
    if 'sampen' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['sampen'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['sampen'], 
                    'brown', marker='o', markersize=3, linewidth=1.5)
            plt.title('Sample Entropy Over Time Windows')
            plt.ylabel('SampEn')
            plt.grid(True, alpha=0.3)
    
    # Plot 8: Approximate Entropy
    ax8 = plt.subplot(6, 2, 8)
    if 'apen' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['apen'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['apen'], 
                    'darkgreen', marker='o', markersize=3, linewidth=1.5)
            plt.title('Approximate Entropy Over Time Windows')
            plt.ylabel('ApEn')
            plt.grid(True, alpha=0.3)
    
    # Plot 9: DFA
    ax9 = plt.subplot(6, 2, 9)
    if 'dfa' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['dfa'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['dfa'], 
                    'magenta', marker='o', markersize=3, linewidth=1.5)
            plt.title('DFA (Fractal Scaling) Over Time Windows')
            plt.ylabel('DFA α')
            plt.grid(True, alpha=0.3)
    
    # Plot 10: Correlation Dimension
    ax10 = plt.subplot(6, 2, 10)
    if 'cd' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['cd'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['cd'], 
                    'cyan', marker='o', markersize=3, linewidth=1.5)
            plt.title('Correlation Dimension Over Time Windows')
            plt.ylabel('CD')
            plt.grid(True, alpha=0.3)
    
    # Plot 11: LF/HF Ratio (STFT)
    ax11 = plt.subplot(6, 2, 11)
    if 'stft_lf_hf_ratio' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['stft_lf_hf_ratio'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['stft_lf_hf_ratio'], 
                    'navy', marker='o', markersize=3, linewidth=1.5)
            plt.title('STFT LF/HF Ratio Over Time Windows')
            plt.ylabel('LF/HF Ratio')
            plt.grid(True, alpha=0.3)
    
    # Plot 12: Multiscale Entropy
    ax12 = plt.subplot(6, 2, 12)
    if 'mse' in hrv_results.columns:
        valid_data = hrv_results.dropna(subset=['mse'])
        if not valid_data.empty:
            plt.plot(valid_data['window_start'], valid_data['mse'], 
                    'darkred', marker='o', markersize=3, linewidth=1.5)
            plt.title('Multiscale Entropy Over Time Windows')
            plt.ylabel('MSE')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()