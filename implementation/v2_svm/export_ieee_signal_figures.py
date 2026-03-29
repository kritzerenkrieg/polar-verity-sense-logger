#!/usr/bin/env python3
"""
Export IEEE-style 300 DPI figures for raw PPG, normalized PPG, and denoised NN intervals.

The script reuses the existing v2_svm preprocessing pipeline to keep the same
channel selection, timestamp parsing, peak detection, and NN denoising logic.
Each exported figure contains three vertically stacked panels suitable for
paper figures:
1. Raw PPG
2. Normalized and bandpass-filtered PPG with detected peaks
3. Denoised NN intervals
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .process_input import (
        bandpass_filter,
        calculate_nn_intervals,
        denoise_nn_intervals,
        discover_input_files,
        highly_sensitive_peak_detection,
        parse_filename,
    )
except ImportError:
    from process_input import (
        bandpass_filter,
        calculate_nn_intervals,
        denoise_nn_intervals,
        discover_input_files,
        highly_sensitive_peak_detection,
        parse_filename,
    )


OUTPUT_DIRECTORY_NAME = "ieee-figures"
IEEE_DPI = 300
IEEE_FIGURE_WIDTH_INCHES = 14.32
IEEE_FIGURE_HEIGHT_INCHES = 5.85


def load_ppg_series(file_path):
    """Load raw PPG and aligned time values from a dataset file."""
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
            raise ValueError("No numeric channels found in the input file.")

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
            ).dt.total_seconds().to_numpy()
        except Exception:
            time = np.arange(len(dataframe), dtype=float)
    else:
        time = np.arange(len(dataframe), dtype=float)

    ppg_raw = dataframe["ppg_raw"].to_numpy(dtype=float)
    valid_mask = ~np.isnan(ppg_raw)
    ppg_raw = ppg_raw[valid_mask]
    time = time[valid_mask] if len(time) == len(valid_mask) else np.arange(len(ppg_raw), dtype=float)

    if len(ppg_raw) == 0:
        raise ValueError("No valid PPG samples found after removing NaN values.")

    return time, ppg_raw


def estimate_sampling_rate(time_values):
    """Estimate the sampling rate from timestamps with a safe fallback."""
    if len(time_values) <= 1:
        return 100.0

    sampling_rate = 1 / np.median(np.diff(time_values))
    if sampling_rate > 1000 or sampling_rate < 1:
        return 100.0
    return float(sampling_rate)


def prepare_signal_views(file_path):
    """Build the raw, normalized, filtered, and denoised views needed for plotting."""
    time, ppg_raw = load_ppg_series(file_path)

    ppg_centered = ppg_raw - np.mean(ppg_raw)
    ppg_normalized = ppg_centered / (np.std(ppg_centered) + 1e-10)
    sampling_rate = estimate_sampling_rate(time)

    ppg_filtered = bandpass_filter(
        ppg_normalized,
        0.5,
        min(5.0, sampling_rate / 2.1),
        sampling_rate,
        order=3,
    )
    ppg_normalized_drift_free = (ppg_filtered - np.mean(ppg_filtered)) / (
        np.std(ppg_filtered) + 1e-10
    )
    peaks, _ = highly_sensitive_peak_detection(ppg_filtered, sampling_rate)
    if len(peaks) <= 1:
        raise ValueError("Insufficient peaks detected for NN interval extraction.")

    nn_intervals_raw, nn_times_raw = calculate_nn_intervals(peaks, time)
    if len(nn_intervals_raw) == 0:
        raise ValueError("No NN intervals detected after physiological filtering.")

    nn_intervals_denoised, nn_times_denoised, artifact_mask = denoise_nn_intervals(
        nn_intervals_raw,
        nn_times_raw,
        mad_threshold=3.0,
        change_threshold=0.2,
        correction_method="interpolation",
    )
    if len(nn_intervals_denoised) == 0:
        raise ValueError("No valid NN intervals remain after denoising.")

    return {
        "time": time,
        "ppg_raw": ppg_raw,
        "ppg_normalized": ppg_normalized,
        "ppg_filtered": ppg_filtered,
        "ppg_normalized_drift_free": ppg_normalized_drift_free,
        "peaks": peaks,
        "nn_intervals_raw": nn_intervals_raw,
        "nn_times_raw": nn_times_raw,
        "nn_intervals_denoised": nn_intervals_denoised,
        "nn_times_denoised": nn_times_denoised,
        "artifact_mask": artifact_mask,
        "sampling_rate": sampling_rate,
    }


def apply_ieee_plot_style():
    """Apply a compact publication-oriented plotting style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 12.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.25,
            "ytick.labelsize": 11.25,
            "legend.fontsize": 11.25,
            "figure.titlesize": 13.75,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.1,
            "savefig.dpi": IEEE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def build_output_path(output_dir, input_path):
    """Create an output filename for the exported figure."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base_name}_ieee_ppg_nn.png")


def export_ieee_figure(file_path, output_dir):
    """Export one three-panel publication figure for a single input file."""
    metadata = parse_filename(os.path.basename(file_path)) or {
        "subject": "unknown",
        "condition": "unknown",
        "label": "unknown",
    }
    signal_views = prepare_signal_views(file_path)

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(IEEE_FIGURE_WIDTH_INCHES, IEEE_FIGURE_HEIGHT_INCHES),
        sharex=False,
        constrained_layout=True,
    )

    axes[0].plot(signal_views["time"], signal_views["ppg_raw"], color="black")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("(a) Raw PPG")
    axes[0].grid(True, alpha=0.2, linewidth=0.4)

    axes[1].plot(
        signal_views["time"],
        signal_views["ppg_normalized_drift_free"],
        color="tab:blue",
        label="Normalized PPG",
    )
    axes[1].plot(
        signal_views["time"][signal_views["peaks"]],
        signal_views["ppg_normalized_drift_free"][signal_views["peaks"]],
        linestyle="none",
        marker="o",
        markersize=2.2,
        color="tab:red",
        label="Detected peaks",
    )
    axes[1].set_ylabel("Z-score")
    axes[1].set_title("(b) Normalized PPG (Drift Removed)")
    axes[1].legend(loc="upper right", frameon=True, borderpad=0.2, handlelength=1.5)
    axes[1].grid(True, alpha=0.2, linewidth=0.4)

    axes[2].plot(
        signal_views["nn_times_denoised"],
        signal_views["nn_intervals_denoised"],
        color="tab:green",
        marker="o",
        markersize=2.0,
        linewidth=0.9,
    )
    axes[2].set_ylabel("NN (ms)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("(c) Denoised NN Intervals")
    axes[2].grid(True, alpha=0.2, linewidth=0.4)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0)

    figure.suptitle(
        f"",
        fontweight="bold",
    )

    output_path = build_output_path(output_dir, file_path)
    figure.savefig(output_path, dpi=IEEE_DPI)
    plt.close(figure)
    return output_path, signal_views


def prompt_for_processing_mode(file_count):
    """Prompt the user to process one file or all files."""
    print("\nProcessing Options:")
    print("0: Select and process a single file")
    print("1: Process all files")

    choice = int(input("Enter your choice (0 or 1): "))
    if choice not in [0, 1]:
        raise ValueError("Choice must be 0 or 1")

    if choice == 0 and file_count > 1:
        file_choice = int(input(f"Enter file number (0-{file_count - 1}): "))
        if file_choice < 0 or file_choice >= file_count:
            raise IndexError("Invalid choice.")
        return choice, file_choice

    return choice, 0


def main():
    apply_ieee_plot_style()

    print("IEEE Figure Export - Raw PPG, Normalized PPG, and Denoised NN Intervals")
    print(f"Figure size: {IEEE_FIGURE_WIDTH_INCHES:.2f} x {IEEE_FIGURE_HEIGHT_INCHES:.2f} in")
    print(f"Output resolution: {IEEE_DPI} DPI")
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
            print(
                f"  {index}: {os.path.relpath(file_path, search_root)} ({file_size:.1f} KB) - [unparseable]"
            )

    try:
        choice, file_choice = prompt_for_processing_mode(len(txt_files))
    except (ValueError, IndexError) as error:
        print(f"Error: {error}")
        return

    output_dir = os.path.join(current_dir, OUTPUT_DIRECTORY_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFigures will be written to: {output_dir}")

    files_to_process = txt_files if choice == 1 else [txt_files[file_choice]]
    successful_files = 0
    failed_files = 0

    for index, file_path in enumerate(files_to_process, start=1):
        print(f"\n[{index}/{len(files_to_process)}] Exporting: {os.path.basename(file_path)}")
        try:
            output_path, signal_views = export_ieee_figure(file_path, output_dir)
            artifact_count = int(np.sum(signal_views["artifact_mask"]))
            print(
                f"Saved: {os.path.relpath(output_path, current_dir)} | "
                f"fs={signal_views['sampling_rate']:.2f} Hz | "
                f"peaks={len(signal_views['peaks'])} | "
                f"denoised NN={len(signal_views['nn_intervals_denoised'])} | "
                f"artifacts corrected={artifact_count}"
            )
            successful_files += 1
        except Exception as error:
            print(f"Failed: {os.path.basename(file_path)} -> {error}")
            failed_files += 1

    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    print(f"Requested files: {len(files_to_process)}")
    print(f"Successfully exported: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()