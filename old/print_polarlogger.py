import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import glob

# Step 1: Find all .txt files
current_dir = os.path.dirname(__file__)
txt_files = glob.glob(os.path.join(current_dir, "*.txt"))

if not txt_files:
    raise FileNotFoundError("No .txt files found in the current directory.")

print("Available .txt files:")
for i, file in enumerate(txt_files):
    print(f"{i}: {os.path.basename(file)}")

choice = int(input("Enter the number of the file to process: "))
if choice < 0 or choice >= len(txt_files):
    raise IndexError("Invalid selection.")

file_path = txt_files[choice]

# Step 2: Load the data
df = pd.read_csv(file_path, sep=';', engine='python')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['Phone timestamp'] = pd.to_datetime(df['Phone timestamp'])

# Step 3: Compute average PPG signal
df['ppg'] = df[['channel 0', 'channel 1', 'channel 2']].mean(axis=1)

# Step 4: Detect peaks in the full signal (not just per 6 ms bin)
signal = df['ppg'].values
time = (df['Phone timestamp'] - df['Phone timestamp'].iloc[0]).dt.total_seconds()

# High and low peaks
high_peaks, _ = find_peaks(signal)
low_peaks, _ = find_peaks(-signal)

# Step 5: Plot
plt.figure(figsize=(12, 6))
plt.plot(time, signal, label='PPG Signal', color='blue')
plt.xlabel("Time (s)")
plt.ylabel("PPG (a.u.)")
plt.title("PPG Signal with High and Low Peaks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
