#!/usr/bin/env python3
"""
PPG Signal Processing - V2 SVM feature extraction.

This variant keeps the original combined time and frequency domain features,
adds non-linear features focused on DFA and Poincare SD1/SD2, and uses
fixed 30-second windows with a 5-second stride.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import butter, filtfilt, find_peaks, periodogram, welch


WINDOW_DURATION_SECONDS = 30
WINDOW_STRIDE_SECONDS = 5
OUTPUT_DIRECTORY_NAME = "output"
MIN_TIME_DOMAIN_INTERVALS = 2
MIN_FREQUENCY_DOMAIN_INTERVALS = 10
MIN_DFA_INTERVALS = 10
MIN_POINCARE_INTERVALS = 3


def parse_filename(filename):
    """Parse filename metadata for either sequence or trimmed dataset naming."""
    base_name = os.path.splitext(filename)[0]
    sequence_pattern = r"pr_([^_]+)_([^_]+)_(truth|lie)-sequence"
    trimmed_pattern = r"pr_([^_]+)_([^_]+)"
    match = re.match(sequence_pattern, base_name)

    if match:
        return {
            "subject": match.group(1),
            "condition": match.group(2),
            "label": match.group(3),
        }

    match = re.match(trimmed_pattern, base_name)
    if match:
        return {
            "subject": match.group(1),
            "condition": match.group(2),
            "label": "unknown",
        }

    return None


def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    """Apply a Butterworth bandpass filter."""
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        if low <= 0 or high >= 1 or low >= high:
            raise ValueError(
                f"Invalid frequency range: {lowcut}-{highcut} Hz for fs={fs} Hz"
            )
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)
    except Exception as error:
        print(f"Error in bandpass filter: {error}")
        return signal


def detect_artifacts_mad(nn_intervals, threshold=3.0):
    """Detect artifacts using median absolute deviation."""
    if len(nn_intervals) < 3:
        return np.zeros(len(nn_intervals), dtype=bool)

    median_nn = np.median(nn_intervals)
    mad = np.median(np.abs(nn_intervals - median_nn))

    if mad == 0:
        mad = np.std(nn_intervals) * 0.6745
        if mad == 0:
            return np.zeros(len(nn_intervals), dtype=bool)

    modified_z_scores = 0.6745 * (nn_intervals - median_nn) / mad
    return np.abs(modified_z_scores) > threshold


def detect_artifacts_percentage_change(nn_intervals, threshold=0.2):
    """Detect artifacts based on interval-to-interval percentage change."""
    if len(nn_intervals) < 2:
        return np.zeros(len(nn_intervals), dtype=bool)

    artifacts = np.zeros(len(nn_intervals), dtype=bool)
    for index in range(1, len(nn_intervals)):
        change = abs(nn_intervals[index] - nn_intervals[index - 1]) / nn_intervals[index - 1]
        if change > threshold:
            artifacts[index] = True
    return artifacts


def correct_artifacts(nn_intervals, nn_times, artifact_mask, method="interpolation"):
    """Correct detected artifacts either by interpolation or removal."""
    if not np.any(artifact_mask):
        return nn_intervals, nn_times

    corrected_nn = nn_intervals.copy()

    if method == "interpolation":
        valid_indices = ~artifact_mask
        if np.sum(valid_indices) < 2:
            print("Warning: Too many artifacts for interpolation. Consider manual inspection.")
            return nn_intervals, nn_times

        valid_times = nn_times[valid_indices]
        valid_nn = nn_intervals[valid_indices]
        interp_func = interp1d(
            valid_times,
            valid_nn,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        corrected_nn[artifact_mask] = interp_func(nn_times[artifact_mask])
    elif method == "remove":
        valid_mask = ~artifact_mask
        corrected_nn = nn_intervals[valid_mask]
        nn_times = nn_times[valid_mask]

    return corrected_nn, nn_times


def denoise_nn_intervals(
    nn_intervals,
    nn_times,
    mad_threshold=3.0,
    change_threshold=0.2,
    correction_method="interpolation",
):
    """Apply robust denoising to NN intervals."""
    if len(nn_intervals) < 3:
        return nn_intervals, nn_times, np.array([])

    artifacts_mad = detect_artifacts_mad(nn_intervals, mad_threshold)
    artifacts_change = detect_artifacts_percentage_change(nn_intervals, change_threshold)
    combined_artifacts = artifacts_mad | artifacts_change

    artifact_percentage = (np.sum(combined_artifacts) / len(nn_intervals)) * 100
    if artifact_percentage > 50:
        correction_method = "remove"

    corrected_nn, corrected_times = correct_artifacts(
        nn_intervals,
        nn_times,
        combined_artifacts,
        correction_method,
    )
    return corrected_nn, corrected_times, combined_artifacts


def calculate_nn_intervals(peaks, time_array):
    """Convert peak locations into physiologically filtered NN intervals."""
    if len(peaks) < 2:
        return np.array([]), np.array([])

    peak_times = time_array[peaks]
    nn_intervals = np.diff(peak_times) * 1000
    valid_mask = (nn_intervals >= 300) & (nn_intervals <= 2000)

    return nn_intervals[valid_mask], peak_times[1:][valid_mask]


def highly_sensitive_peak_detection(signal, fs, min_hr=40, max_hr=180):
    """Detect peaks with permissive settings for wearable PPG signals."""
    min_distance = max(int(fs * 60 / max_hr), 1)
    prominence_threshold = 0.01 * np.std(signal)
    height_threshold = np.min(signal) - 1e-5

    peaks, properties = find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence_threshold,
        height=height_threshold,
        width=1,
    )
    valid_peak_mask = signal[peaks] > 0
    return peaks[valid_peak_mask], properties


def calculate_hrv_time_domain_metrics_short(nn_intervals):
    """Calculate short-window time-domain HRV metrics."""
    if len(nn_intervals) < MIN_TIME_DOMAIN_INTERVALS:
        return None

    nn = np.asarray(nn_intervals)
    successive_diffs = np.diff(nn)

    nn_min = np.min(nn)
    nn_max = np.max(nn)

    try:
        hist_counts = np.histogram(nn.astype(int), bins=range(int(nn_min), int(nn_max) + 2))[0]
        triangular_index = len(nn) / np.max(hist_counts) if np.max(hist_counts) > 0 else np.nan
    except Exception:
        triangular_index = np.nan

    return {
        "nn_count": len(nn),
        "nn_mean": np.mean(nn),
        "nn_min": nn_min,
        "nn_max": nn_max,
        "sdnn": np.std(nn, ddof=1) if len(nn) > 1 else np.nan,
        "sdsd": np.std(successive_diffs, ddof=1) if len(successive_diffs) > 1 else np.nan,
        "rmssd": np.sqrt(np.mean(successive_diffs ** 2)) if len(successive_diffs) > 0 else np.nan,
        "pnn20": (
            np.sum(np.abs(successive_diffs) > 20) / len(successive_diffs) * 100
            if len(successive_diffs) > 0
            else np.nan
        ),
        "pnn50": (
            np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100
            if len(successive_diffs) > 0
            else np.nan
        ),
        "triangular_index": triangular_index,
    }


def interpolate_nn_intervals(nn_intervals, nn_times, fs_target=4.0):
    """Interpolate irregular NN intervals to an evenly sampled series."""
    if len(nn_intervals) < 3:
        return None, None

    time_interp = np.arange(nn_times[0], nn_times[-1], 1 / fs_target)
    if len(time_interp) < 4:
        return None, None

    try:
        if len(nn_intervals) >= 4:
            spline = UnivariateSpline(nn_times, nn_intervals, s=0)
            nn_interp = spline(time_interp)
        else:
            interp_func = interp1d(
                nn_times,
                nn_intervals,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            nn_interp = interp_func(time_interp)
        return nn_interp, time_interp
    except Exception as error:
        print(f"Interpolation error: {error}")
        return None, None


def calculate_psd_welch(nn_interp, fs, nperseg=None):
    """Estimate PSD using Welch's method."""
    if nperseg is None:
        nperseg = min(len(nn_interp) // 2, int(fs * 60))
        nperseg = max(nperseg, int(fs * 10))

    try:
        nn_detrended = nn_interp - np.mean(nn_interp)
        noverlap = int(nperseg * 0.5)
        freq, psd = welch(
            nn_detrended,
            fs=fs,
            nperseg=nperseg,
            window="hann",
            noverlap=noverlap,
            detrend="linear",
        )
        return freq, psd
    except Exception as error:
        print(f"PSD calculation error: {error}")
        return None, None


def calculate_hrv_frequency_domain_metrics(nn_intervals, nn_times, method="welch"):
    """Calculate standard LF and HF frequency-domain HRV metrics."""
    if len(nn_intervals) < MIN_FREQUENCY_DOMAIN_INTERVALS:
        return None

    fs_target = 4.0
    nn_interp, _ = interpolate_nn_intervals(nn_intervals, nn_times, fs_target)
    if nn_interp is None:
        return None

    if method == "welch":
        freq, psd = calculate_psd_welch(nn_interp, fs_target)
    else:
        try:
            nn_detrended = nn_interp - np.mean(nn_interp)
            freq, psd = periodogram(
                nn_detrended,
                fs=fs_target,
                window="hann",
                detrend="linear",
            )
        except Exception as error:
            print(f"Periodogram calculation error: {error}")
            return None

    if freq is None or psd is None:
        return None

    lf_idx = (freq >= 0.04) & (freq < 0.15)
    hf_idx = (freq >= 0.15) & (freq < 0.4)

    lf_power = trapezoid(psd[lf_idx], freq[lf_idx]) if np.any(lf_idx) else 0
    hf_power = trapezoid(psd[hf_idx], freq[hf_idx]) if np.any(hf_idx) else 0
    total_power = lf_power + hf_power

    return {
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_power / hf_power if hf_power > 0 else np.nan,
        "lf_norm": (lf_power / total_power) * 100 if total_power > 0 else np.nan,
        "hf_norm": (hf_power / total_power) * 100 if total_power > 0 else np.nan,
        "ln_hf": np.log(hf_power) if hf_power > 0 else np.nan,
        "lf_peak": freq[lf_idx][np.argmax(psd[lf_idx])] if np.any(lf_idx) else np.nan,
        "hf_peak": freq[hf_idx][np.argmax(psd[hf_idx])] if np.any(hf_idx) else np.nan,
        "n_samples": len(nn_interp),
    }


def calculate_poincare_features(nn_intervals):
    """Calculate Poincare SD1, SD2, and the SD1/SD2 ratio."""
    if len(nn_intervals) < MIN_POINCARE_INTERVALS:
        return {
            "sd1": np.nan,
            "sd2": np.nan,
            "sd1_sd2_ratio": np.nan,
        }

    nn = np.asarray(nn_intervals)
    x_values = nn[:-1]
    y_values = nn[1:]
    if len(x_values) < 2:
        return {
            "sd1": np.nan,
            "sd2": np.nan,
            "sd1_sd2_ratio": np.nan,
        }

    diff_x_y = x_values - y_values
    sum_x_y = x_values + y_values

    sd1 = np.sqrt(np.var(diff_x_y, ddof=1) / 2.0) if len(diff_x_y) > 1 else np.nan
    sd2_term = 2 * np.var(sum_x_y, ddof=1) / 4.0 - np.var(diff_x_y, ddof=1) / 4.0
    sd2 = np.sqrt(sd2_term) if sd2_term > 0 else np.nan

    return {
        "sd1": sd1,
        "sd2": sd2,
        "sd1_sd2_ratio": sd1 / sd2 if sd2 and np.isfinite(sd2) and sd2 > 0 else np.nan,
    }


def calculate_dfa(nn_intervals, min_window=4, max_window=None):
    """Calculate the detrended fluctuation analysis scaling exponent."""
    if len(nn_intervals) < MIN_DFA_INTERVALS:
        return np.nan

    data = np.asarray(nn_intervals, dtype=float)
    n_points = len(data)
    if max_window is None:
        max_window = max(min(n_points // 4, 16), min_window + 2)

    if max_window <= min_window:
        return np.nan

    integrated = np.cumsum(data - np.mean(data))
    windows = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=min(12, max_window - min_window + 1)).astype(int)
    )

    fluctuations = []
    valid_windows = []
    for window_size in windows:
        if window_size < 2:
            continue

        n_segments = n_points // window_size
        if n_segments < 2:
            continue

        rms_values = []
        for segment_index in range(n_segments):
            start = segment_index * window_size
            end = start + window_size
            segment = integrated[start:end]
            x_values = np.arange(window_size)
            coefficients = np.polyfit(x_values, segment, 1)
            trend = np.polyval(coefficients, x_values)
            rms_values.append(np.sqrt(np.mean((segment - trend) ** 2)))

        mean_rms = np.mean(rms_values)
        if np.isfinite(mean_rms) and mean_rms > 0:
            valid_windows.append(window_size)
            fluctuations.append(mean_rms)

    if len(fluctuations) < 3:
        return np.nan

    coefficients = np.polyfit(np.log10(valid_windows), np.log10(fluctuations), 1)
    return coefficients[0]


def calculate_nonlinear_metrics(nn_intervals):
    """Calculate the non-linear features used in the v2 pipeline."""
    metrics = {
        "sd1": np.nan,
        "sd2": np.nan,
        "sd1_sd2_ratio": np.nan,
        "dfa": np.nan,
    }

    metrics.update(calculate_poincare_features(nn_intervals))
    metrics["dfa"] = calculate_dfa(nn_intervals)
    return metrics


def extract_window_nn_intervals(nn_intervals, nn_times, start_time, window_duration=WINDOW_DURATION_SECONDS):
    """Extract NN intervals contained in a specific window."""
    end_time = start_time + window_duration
    nn_intervals = np.asarray(nn_intervals)
    nn_times = np.asarray(nn_times)
    window_mask = (nn_times >= start_time) & (nn_times < end_time)
    return nn_intervals[window_mask], nn_times[window_mask]


def build_window_starts(total_duration, window_duration=WINDOW_DURATION_SECONDS, stride=WINDOW_STRIDE_SECONDS):
    """Build full windows over the recording using a fixed stride."""
    if total_duration < window_duration:
        return np.array([])
    return np.arange(0, total_duration - window_duration + 1e-9, stride)


def sliding_window_hrv_analysis_v2(nn_intervals, nn_times):
    """Run combined time, frequency, and non-linear analysis using 30 s / 5 s windows."""
    if len(nn_intervals) == 0 or len(nn_times) == 0:
        return pd.DataFrame()

    nn_intervals = np.asarray(nn_intervals)
    nn_times = np.asarray(nn_times)
    recording_start = nn_times[0]
    total_duration = nn_times[-1] - recording_start
    window_starts = recording_start + build_window_starts(total_duration)

    print(f"Total recording duration: {total_duration:.1f} seconds")
    print(f"Window duration: {WINDOW_DURATION_SECONDS} seconds")
    print(f"Window stride: {WINDOW_STRIDE_SECONDS} seconds")
    print(f"Number of windows: {len(window_starts)}")

    results = []
    for index, start_time in enumerate(window_starts, start=1):
        window_nn, window_times = extract_window_nn_intervals(nn_intervals, nn_times, start_time)
        relative_start = start_time - recording_start

        combined_metrics = {
            "window_start": relative_start,
            "window_end": relative_start + WINDOW_DURATION_SECONDS,
            "window_number": index,
            "nn_count": len(window_nn),
        }

        time_metrics = calculate_hrv_time_domain_metrics_short(window_nn)
        if time_metrics is not None:
            for key, value in time_metrics.items():
                if key not in combined_metrics:
                    combined_metrics[key] = value

        if "sdnn" not in combined_metrics:
            combined_metrics.update(
                {
                    "nn_mean": np.nan,
                    "nn_min": np.nan,
                    "nn_max": np.nan,
                    "sdnn": np.nan,
                    "sdsd": np.nan,
                    "rmssd": np.nan,
                    "pnn20": np.nan,
                    "pnn50": np.nan,
                    "triangular_index": np.nan,
                }
            )

        freq_metrics = calculate_hrv_frequency_domain_metrics(window_nn, window_times)
        if freq_metrics is not None:
            combined_metrics.update(freq_metrics)
        else:
            combined_metrics.update(
                {
                    "lf_power": np.nan,
                    "hf_power": np.nan,
                    "lf_hf_ratio": np.nan,
                    "lf_norm": np.nan,
                    "hf_norm": np.nan,
                    "ln_hf": np.nan,
                    "lf_peak": np.nan,
                    "hf_peak": np.nan,
                    "n_samples": 0,
                }
            )

        combined_metrics.update(calculate_nonlinear_metrics(window_nn))
        results.append(combined_metrics)

    df_results = pd.DataFrame(results)
    print(f"Final output: {len(df_results)} windows")
    print(f"Combined features: {len(df_results.columns)} total columns")
    return df_results


def process_single_file(file_path, show_plots=True):
    """Process a single PPG file and return v2 combined HRV features."""
    print(f"\nProcessing file: {os.path.basename(file_path)}")
    print("-" * 50)

    metadata = parse_filename(os.path.basename(file_path))
    if not metadata:
        print("Warning: Could not parse filename for metadata")
        metadata = {"subject": "unknown", "condition": "unknown", "label": "unknown"}

    print(
        f"Subject: {metadata['subject']}, Condition: {metadata['condition']}, Label: {metadata['label']}"
    )

    try:
        try:
            dataframe = pd.read_csv(file_path, sep=";", engine="python")
        except Exception:
            dataframe = pd.read_csv(file_path, sep=",", engine="python")

        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]

        required_channels = ["channel 0", "channel 1", "channel 2"]
        available_channels = [column for column in required_channels if column in dataframe.columns]
        if not available_channels:
            numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                available_channels = numeric_columns[:3]
            else:
                print("No numeric channels found.")
                return None, file_path

        if len(available_channels) > 1:
            dataframe["ppg_raw"] = dataframe[available_channels].mean(axis=1)
        else:
            dataframe["ppg_raw"] = dataframe[available_channels[0]]

        timestamp_columns = [
            column
            for column in dataframe.columns
            if "timestamp" in column.lower() or "time" in column.lower()
        ]
        if timestamp_columns:
            try:
                dataframe["Phone timestamp"] = pd.to_datetime(dataframe[timestamp_columns[0]])
                time = (
                    dataframe["Phone timestamp"] - dataframe["Phone timestamp"].iloc[0]
                ).dt.total_seconds()
            except Exception:
                time = np.arange(len(dataframe))
        else:
            time = np.arange(len(dataframe))

        ppg = dataframe["ppg_raw"]
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

        ppg_filtered = bandpass_filter(ppg_scaled.values, 0.5, min(5.0, fs / 2.1), fs, order=3)
        peaks, _ = highly_sensitive_peak_detection(ppg_filtered, fs)
        print(f"Detected {len(peaks)} peaks")

        if len(peaks) <= 1:
            print("Insufficient peaks detected")
            return None, file_path

        nn_intervals_raw, nn_times_raw = calculate_nn_intervals(peaks, time)
        if len(nn_intervals_raw) == 0:
            print("No NN intervals detected")
            return None, file_path

        print("Denoising NN intervals...")
        nn_intervals, nn_times, _ = denoise_nn_intervals(
            nn_intervals_raw,
            nn_times_raw,
            mad_threshold=3.0,
            change_threshold=0.2,
            correction_method="interpolation",
        )

        print(f"Original NN intervals: {len(nn_intervals_raw)}")
        print(f"Final NN intervals after denoising: {len(nn_intervals)}")

        if len(nn_intervals) == 0:
            print("No valid NN intervals after denoising")
            return None, file_path

        print("Performing v2 sliding window analysis (30s windows, 5s stride)...")
        combined_results = sliding_window_hrv_analysis_v2(nn_intervals, nn_times)
        if combined_results.empty:
            print("No valid windows found for analysis")
            return None, file_path

        combined_results["subject"] = metadata["subject"]
        combined_results["condition"] = metadata["condition"]
        combined_results["label"] = metadata["label"]

        print(
            f"Successfully analyzed {len(combined_results)} windows with {len(combined_results.columns)} combined features"
        )

        if show_plots:
            create_combined_plots(
                combined_results,
                time,
                ppg_filtered,
                peaks,
                nn_intervals,
                nn_times,
                metadata,
            )

        return combined_results, file_path
    except Exception as error:
        print(f"Error processing {os.path.basename(file_path)}: {error}")
        import traceback

        traceback.print_exc()
        return None, file_path


def create_combined_plots(combined_results, time, ppg_filtered, peaks, nn_intervals, nn_times, metadata):
    """Create visualization for the combined v2 feature set."""
    figure = plt.figure(figsize=(20, 24))
    figure.suptitle(
        f"Combined HRV Analysis V2 - {metadata['subject']} {metadata['condition']} {metadata['label']}",
        fontsize=16,
        fontweight="bold",
    )

    plt.subplot(6, 2, 1)
    plt.plot(time, ppg_filtered, label="Filtered PPG", color="blue", alpha=0.7)
    if len(peaks) > 0:
        plt.plot(time[peaks], ppg_filtered[peaks], "ro", markersize=3, label=f"Peaks ({len(peaks)})")
    plt.title("PPG Signal with Detected Peaks")
    plt.ylabel("Normalized PPG")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(6, 2, 2)
    if len(nn_intervals) > 0:
        plt.plot(nn_times, nn_intervals, "g-o", markersize=2, linewidth=1, label="NN Intervals")
        plt.axhline(y=np.mean(nn_intervals), color="red", linestyle="--", label=f"Mean: {np.mean(nn_intervals):.1f} ms")
        plt.ylim(300, 2000)
    plt.title("NN Intervals Over Time")
    plt.ylabel("NN Interval (ms)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_specs = [
        ("sdnn", "Time Domain: SDNN", "SDNN (ms)", "b-o"),
        ("rmssd", "Time Domain: RMSSD", "RMSSD (ms)", "r-o"),
        ("lf_hf_ratio", "Frequency Domain: LF/HF Ratio", "LF/HF Ratio", "m-o"),
        ("ln_hf", "Frequency Domain: ln(HF)", "ln(HF)", "g-o"),
        ("sd1_sd2_ratio", "Non-Linear: SD1/SD2", "SD1/SD2", "c-o"),
        ("dfa", "Non-Linear: DFA", "DFA", "k-o"),
        ("pnn50", "Time Domain: pNN50", "pNN50 (%)", "y-o"),
        ("lf_power", "Frequency Domain: LF Power", "LF Power (ms²)", "b-o"),
        ("hf_power", "Frequency Domain: HF Power", "HF Power (ms²)", "r-o"),
        ("nn_mean", "Heart Rate Over Time", "Heart Rate (BPM)", "m-o"),
    ]

    for subplot_index, (column, title, ylabel, style) in enumerate(plot_specs, start=3):
        plt.subplot(6, 2, subplot_index)
        valid_rows = combined_results.dropna(subset=[column])
        if not valid_rows.empty:
            if column == "nn_mean":
                heart_rate = 60000 / valid_rows[column]
                plt.plot(valid_rows["window_start"], heart_rate, style, markersize=3, linewidth=1.5)
            else:
                plt.plot(valid_rows["window_start"], valid_rows[column], style, markersize=3, linewidth=1.5)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Time (s)")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def display_combined_summary_statistics(combined_results):
    """Print summary statistics for the extracted features."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS ACROSS ALL WINDOWS - V2 SVM FEATURES")
    print("=" * 80)
    print(f"Total windows analyzed: {len(combined_results)}")
    print(f"Total features per window: {len(combined_results.columns)}")

    metrics = [
        "sdnn",
        "rmssd",
        "pnn50",
        "lf_power",
        "hf_power",
        "lf_hf_ratio",
        "sd1_sd2_ratio",
        "dfa",
    ]
    for metric in metrics:
        if metric not in combined_results.columns:
            continue

        valid_values = combined_results[metric].dropna()
        if len(valid_values) == 0:
            continue

        print(f"\n{metric.upper()}:")
        print(f"  Mean: {valid_values.mean():.4f}")
        print(f"  Std:  {valid_values.std():.4f}")
        print(f"  Min:  {valid_values.min():.4f}")
        print(f"  Max:  {valid_values.max():.4f}")


def discover_input_files(current_dir):
    """Discover segmented dataset files from datasets/segmented."""
    repo_root = os.path.dirname(os.path.dirname(current_dir))
    segmented_root = os.path.join(repo_root, "datasets", "segmented")
    txt_files = []

    if os.path.isdir(segmented_root):
        txt_files.extend(glob.glob(os.path.join(segmented_root, "pr_*_*_*-sequence.txt")))

    return sorted(set(txt_files)), segmented_root


def main():
    print("PPG Signal Processing - V2 SVM Combined HRV Analysis")
    print("30-second windows moving 5 seconds forward")
    print("Includes DFA and Poincare SD1/SD2 non-linear features")
    print("=" * 80)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    txt_files, search_root = discover_input_files(current_dir)
    print(f"Scanning directory: {search_root}")

    if not txt_files:
        print("No .txt files found.")
        return

    print(f"Found {len(txt_files)} .txt file(s):")
    for index, file_path in enumerate(txt_files):
        file_size = os.path.getsize(file_path) / 1024
        metadata = parse_filename(os.path.basename(file_path))
        if metadata:
            print(
                f"  {index}: {os.path.relpath(file_path, search_root)} ({file_size:.1f} KB) - "
                f"{metadata['subject']} {metadata['condition']} {metadata['label']}"
            )
        else:
            print(f"  {index}: {os.path.relpath(file_path, search_root)} ({file_size:.1f} KB) - [unparseable]")

    print("\nProcessing Options:")
    print("0: Select and process a single file")
    print("1: Process all files")

    try:
        choice = int(input("Enter your choice (0 or 1): "))
        if choice not in [0, 1]:
            raise ValueError("Choice must be 0 or 1")
    except ValueError as error:
        print(f"Error: {error}")
        return

    output_dir = os.path.join(current_dir, OUTPUT_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput CSVs will be written to: {output_dir}")

    if choice == 0:
        if len(txt_files) == 1:
            file_choice = 0
            print(f"Automatically selecting: {os.path.basename(txt_files[0])}")
        else:
            try:
                file_choice = int(input(f"Enter file number (0-{len(txt_files) - 1}): "))
                if file_choice < 0 or file_choice >= len(txt_files):
                    raise IndexError("Invalid choice.")
            except (ValueError, IndexError) as error:
                print(f"Error: {error}")
                return

        file_path = txt_files[file_choice]
        combined_results, processed_file = process_single_file(file_path, show_plots=True)
        if combined_results is None:
            print("Failed to process the selected file.")
            return

        base_name = os.path.splitext(os.path.basename(processed_file))[0]
        output_filename = os.path.join(output_dir, f"{base_name}_combined_hrv_v2_svm.csv")
        combined_results.to_csv(output_filename, index=False)
        print(f"\nCombined HRV results saved to: {output_filename}")
        print(f"Output contains {len(combined_results)} windows")
        print(f"Total features per sample: {len(combined_results.columns)}")
        display_combined_summary_statistics(combined_results)
        return

    print(f"\nProcessing all {len(txt_files)} files...")
    successful_files = 0
    failed_files = 0

    for index, file_path in enumerate(txt_files, start=1):
        print(f"\n[{index}/{len(txt_files)}] Processing: {os.path.basename(file_path)}")
        combined_results, processed_file = process_single_file(file_path, show_plots=False)

        if combined_results is None:
            print(f"Failed to process: {os.path.basename(processed_file)}")
            failed_files += 1
            continue

        base_name = os.path.splitext(os.path.basename(processed_file))[0]
        output_filename = os.path.join(output_dir, f"{base_name}_combined_hrv_v2_svm.csv")
        combined_results.to_csv(output_filename, index=False)
        print(
            f"Saved: {os.path.relpath(output_filename, current_dir)} "
            f"({len(combined_results)} windows, {len(combined_results.columns)} features)"
        )
        successful_files += 1

    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(txt_files)}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Windowing: {WINDOW_DURATION_SECONDS}s with {WINDOW_STRIDE_SECONDS}s stride")
    print("Features: combined time, frequency, DFA, and SD1/SD2")


if __name__ == "__main__":
    main()