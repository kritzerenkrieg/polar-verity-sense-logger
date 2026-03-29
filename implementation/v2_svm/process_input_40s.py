#!/usr/bin/env python3
"""
PPG Signal Processing - V2 SVM feature extraction for the last 40 seconds.

This variant reuses the combined time, frequency, and non-linear feature
pipeline from process_input.py, but limits analysis to the last 40 seconds
of each recording using fixed 30-second windows with a 1-second stride.
That yields exactly 11 windows per file when the recording is long enough.
"""

import os

import numpy as np
import pandas as pd

try:
    from .process_input import (
        bandpass_filter,
        calculate_hrv_frequency_domain_metrics,
        calculate_hrv_time_domain_metrics_short,
        calculate_nn_intervals,
        calculate_nonlinear_metrics,
        create_combined_plots,
        denoise_nn_intervals,
        discover_input_files,
        display_combined_summary_statistics,
        extract_window_nn_intervals,
        highly_sensitive_peak_detection,
        parse_filename,
    )
except ImportError:
    from process_input import (
        bandpass_filter,
        calculate_hrv_frequency_domain_metrics,
        calculate_hrv_time_domain_metrics_short,
        calculate_nn_intervals,
        calculate_nonlinear_metrics,
        create_combined_plots,
        denoise_nn_intervals,
        discover_input_files,
        display_combined_summary_statistics,
        extract_window_nn_intervals,
        highly_sensitive_peak_detection,
        parse_filename,
    )


WINDOW_DURATION_SECONDS = 30
WINDOW_STRIDE_SECONDS = 1
LAST_SECONDS = 40
EXPECTED_WINDOWS = 11
OUTPUT_DIRECTORY_NAME = "40s-output"


def build_last40_window_starts(nn_times):
    """Build absolute window start times over the last 40 seconds only."""
    if len(nn_times) == 0:
        return np.array([])

    recording_start = nn_times[0]
    recording_end = nn_times[-1]
    total_duration = recording_end - recording_start
    if total_duration < LAST_SECONDS:
        return np.array([])

    analysis_start = recording_end - LAST_SECONDS
    return analysis_start + np.arange(EXPECTED_WINDOWS) * WINDOW_STRIDE_SECONDS


def sliding_window_hrv_analysis_v2_last40_seconds(nn_intervals, nn_times):
    """Run combined analysis over the last 40 seconds using 30 s / 1 s windows."""
    if len(nn_intervals) == 0 or len(nn_times) == 0:
        return pd.DataFrame()

    nn_intervals = np.asarray(nn_intervals)
    nn_times = np.asarray(nn_times)
    recording_start = nn_times[0]
    recording_end = nn_times[-1]
    total_duration = recording_end - recording_start
    window_starts = build_last40_window_starts(nn_times)

    print(f"Total recording duration: {total_duration:.1f} seconds")
    print(
        f"Analysis period: last {LAST_SECONDS} seconds "
        f"({recording_end - LAST_SECONDS - recording_start:.1f}-{total_duration:.1f}s)"
    )
    print(f"Window duration: {WINDOW_DURATION_SECONDS} seconds")
    print(f"Window stride: {WINDOW_STRIDE_SECONDS} second")
    print(f"Number of windows: {len(window_starts)}")

    if len(window_starts) == 0:
        print(
            f"Recording is too short for last-{LAST_SECONDS}-seconds analysis "
            f"(requires at least {LAST_SECONDS} seconds)."
        )
        return pd.DataFrame()

    results = []
    for index, start_time in enumerate(window_starts, start=1):
        window_nn, window_times = extract_window_nn_intervals(
            nn_intervals,
            nn_times,
            start_time,
            WINDOW_DURATION_SECONDS,
        )
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
    print(f"Final output: {len(df_results)} windows (expected: {EXPECTED_WINDOWS})")
    print(f"Combined features: {len(df_results.columns)} total columns")
    return df_results


def process_single_file(file_path, show_plots=True):
    """Process a single PPG file and return 40-second combined HRV features."""
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

        print(
            "Performing 40-second v2 sliding window analysis "
            "(last 40 seconds, 30s windows, 1s stride)..."
        )
        combined_results = sliding_window_hrv_analysis_v2_last40_seconds(nn_intervals, nn_times)
        if combined_results.empty:
            print("No valid windows found for analysis")
            return None, file_path

        combined_results["subject"] = metadata["subject"]
        combined_results["condition"] = metadata["condition"]
        combined_results["label"] = metadata["label"]

        print(
            f"Successfully analyzed {len(combined_results)} windows with "
            f"{len(combined_results.columns)} combined features"
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


def main():
    print("PPG Signal Processing - V2 SVM Combined HRV Analysis (Last 40 Seconds)")
    print("30-second windows moving 1 second forward over the last 40 seconds")
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
        output_filename = os.path.join(output_dir, f"{base_name}_combined_hrv_v2_svm_40s.csv")
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
        output_filename = os.path.join(output_dir, f"{base_name}_combined_hrv_v2_svm_40s.csv")
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
    print(f"Windowing: last {LAST_SECONDS}s, {WINDOW_DURATION_SECONDS}s windows, {WINDOW_STRIDE_SECONDS}s stride")
    print("Features: combined time, frequency, DFA, and SD1/SD2")


if __name__ == "__main__":
    main()