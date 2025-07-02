import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

def extract_sequences(file_path, output_dir):
    """
    Extract truth and lie sequences from lie detection data
    
    Sequence pattern (30 seconds total per cycle):
    - 5s audio cue + 5s text cue + 10s answer + 10s neutral = 30s
    - Pattern: Truth, Lie, Truth, Lie, etc.
    - Extract only the active portions (20s), skip neutral (10s)
    """
    
    try:
        # Read the CSV file
        print(f"Processing: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path, sep=';', engine='python')
        
        # Parse timestamps
        df['Phone timestamp'] = pd.to_datetime(df['Phone timestamp'])
        start_time = df['Phone timestamp'].iloc[0]
        
        # Calculate relative time in seconds
        df['relative_time'] = (df['Phone timestamp'] - start_time).dt.total_seconds()
        
        # Get total duration
        total_duration = df['relative_time'].iloc[-1]
        print(f"Total duration: {total_duration:.1f} seconds")
        
        # Calculate sampling rate
        time_diffs = np.diff(df['relative_time'])
        sampling_rate = 1 / np.median(time_diffs)
        print(f"Estimated sampling rate: {sampling_rate:.1f} Hz")
        
        # Sequence timing parameters
        cycle_duration = 30.0  # seconds per cycle (20s active + 10s neutral)
        active_duration = 20.0  # seconds of active data (cue + answer)
        neutral_duration = 10.0  # seconds of neutral/rest
        
        # Calculate number of complete cycles (full 30s cycles)
        num_complete_cycles = int(total_duration // cycle_duration)
        
        # Check if there's an incomplete cycle with at least 20s of active data
        remaining_time = total_duration - (num_complete_cycles * cycle_duration)
        has_partial_cycle = remaining_time >= active_duration
        
        total_cycles = num_complete_cycles + (1 if has_partial_cycle else 0)
        
        print(f"Complete cycles (30s): {num_complete_cycles}")
        if has_partial_cycle:
            print(f"Partial cycle with active data (20s): 1")
        print(f"Total cycles to process: {total_cycles}")
        
        if total_cycles == 0:
            print("Warning: Data too short for even one cycle with 20s active data")
            return False
        
        # Initialize lists for truth and lie sequences
        truth_sequences = []
        lie_sequences = []
        
        # Extract sequences
        for cycle in range(total_cycles):
            cycle_start = cycle * cycle_duration
            active_start = cycle_start
            active_end = cycle_start + active_duration
            
            # For the last cycle, make sure we don't go beyond available data
            if active_end > total_duration:
                active_end = total_duration
                print(f"Cycle {cycle + 1}: {active_start:.1f}s - {active_end:.1f}s (partial)", end="")
            else:
                print(f"Cycle {cycle + 1}: {active_start:.1f}s - {active_end:.1f}s", end="")
            
            # Extract active portion of this cycle
            mask = (df['relative_time'] >= active_start) & (df['relative_time'] < active_end)
            cycle_data = df[mask].copy()
            
            if len(cycle_data) == 0:
                print(" - No data found")
                continue
            
            # Reset timestamps for this sequence (start from 0)
            cycle_data['relative_time'] = cycle_data['relative_time'] - active_start
            
            # Determine if this is truth or lie based on cycle number
            # Cycle 0, 2, 4, ... = Truth
            # Cycle 1, 3, 5, ... = Lie
            if cycle % 2 == 0:
                truth_sequences.append(cycle_data)
                print(" - TRUTH")
            else:
                lie_sequences.append(cycle_data)
                print(" - LIE")
        
        # Combine all truth and lie sequences
        if truth_sequences:
            combined_truth = pd.concat(truth_sequences, ignore_index=True)
            # Recalculate continuous timestamps with current processing date
            combined_truth = recalculate_timestamps(combined_truth)
        else:
            combined_truth = None
            
        if lie_sequences:
            combined_lie = pd.concat(lie_sequences, ignore_index=True)
            # Recalculate continuous timestamps with current processing date
            combined_lie = recalculate_timestamps(combined_lie)
        else:
            combined_lie = None
        
        # Generate output filenames
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        truth_filename = f"{base_filename}_truth-sequence.txt"
        lie_filename = f"{base_filename}_lie-sequence.txt"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save truth sequence
        if combined_truth is not None:
            truth_path = os.path.join(output_dir, truth_filename)
            # Remove the relative_time column before saving
            output_truth = combined_truth.drop('relative_time', axis=1)
            output_truth.to_csv(truth_path, sep=';', index=False)
            print(f"Truth sequence saved: {truth_path} ({len(combined_truth)} samples)")
        else:
            print("No truth sequences found")
        
        # Save lie sequence
        if combined_lie is not None:
            lie_path = os.path.join(output_dir, lie_filename)
            # Remove the relative_time column before saving
            output_lie = combined_lie.drop('relative_time', axis=1)
            output_lie.to_csv(lie_path, sep=';', index=False)
            print(f"Lie sequence saved: {lie_path} ({len(combined_lie)} samples)")
        else:
            print("No lie sequences found")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def recalculate_timestamps(df):
    """
    Recalculate timestamps to create continuous sequence using current processing date,
    starting from midnight (00:00:00).
    """
    # Calculate time intervals between samples
    time_intervals = np.diff(df['relative_time'])
    median_interval = np.median(time_intervals)
    
    # Create new continuous relative times
    new_relative_times = np.arange(len(df)) * median_interval
    
    # Update the dataframe
    df = df.copy()
    df['relative_time'] = new_relative_times
    
    # Use current date at 00:00:00 as base timestamp
    processing_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    new_timestamps = [processing_date + timedelta(seconds=t) for t in new_relative_times]
    df['Phone timestamp'] = new_timestamps
    
    return df

def main():
    print("Lie Detection Sequence Extractor")
    print("=" * 50)
    print("Sequence Pattern:")
    print("- 5s audio cue + 5s text cue + 10s answer + 10s neutral = 30s cycle")
    print("- Truth cycles: 0, 2, 4, 6, ...")
    print("- Lie cycles: 1, 3, 5, 7, ...")
    print("- Extracts only active portions (20s), skips neutral (10s)")
    print("- Timestamps will be set to current processing date/time")
    print("=" * 50)
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    print(f"Scanning directory: {current_dir}")

    # Search for text files
    txt_files = glob.glob(os.path.join(current_dir, "**/*.txt"), recursive=True)
    txt_files.extend(glob.glob(os.path.join(current_dir, "*.txt")))
    
    # Filter out files that are already processed sequences
    txt_files = [f for f in txt_files if not (f.endswith('_truth-sequence.txt') or f.endswith('_lie-sequence.txt'))]
    txt_files = sorted(list(set(txt_files)))

    if not txt_files:
        print("No .txt files found.")
        return

    print(f"Found {len(txt_files)} .txt file(s):")
    for i, file in enumerate(txt_files):
        file_size = os.path.getsize(file) / 1024
        print(f"{i}: {os.path.relpath(file, current_dir)} ({file_size:.1f} KB)")

    # Setup output directory
    output_dir = os.path.join(current_dir, "datasets", "sequenced")
    print(f"Output directory: {output_dir}")

    # Ask user for processing mode
    print("\nProcessing options:")
    print("0: Process all files automatically")
    print("1: Select specific file")
    
    try:
        mode = input("Enter choice (0 or 1): ").strip()
        
        if mode == "0":
            # Process all files
            success_count = 0
            processing_start_time = datetime.now()
            print(f"Processing started at: {processing_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            for file_path in txt_files:
                print(f"\n{'-'*50}")
                if extract_sequences(file_path, output_dir):
                    success_count += 1
            
            print(f"\n{'='*50}")
            print(f"Processing complete!")
            print(f"Successfully processed: {success_count}/{len(txt_files)} files")
            print(f"Output directory: {output_dir}")
            print(f"All sequences timestamped with processing date: {processing_start_time.strftime('%Y-%m-%d')}")
            
        elif mode == "1":
            # Select specific file
            choice = int(input(f"Enter file number (0-{len(txt_files)-1}): "))
            if choice < 0 or choice >= len(txt_files):
                raise IndexError("Invalid choice.")
            
            file_path = txt_files[choice]
            processing_time = datetime.now()
            print(f"Processing at: {processing_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\n{'-'*50}")
            if extract_sequences(file_path, output_dir):
                print(f"Successfully processed: {os.path.basename(file_path)}")
                print(f"Sequences timestamped with: {processing_time.strftime('%Y-%m-%d')}")
            else:
                print(f"Failed to process: {os.path.basename(file_path)}")
        
        else:
            print("Invalid choice.")
            return
            
    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        return

    print(f"\nOutput files saved to: {output_dir}")

if __name__ == "__main__":
    main()