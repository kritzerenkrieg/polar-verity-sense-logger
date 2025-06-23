import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os
import glob

# Fungsi Bandpass Butterworth filter
def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Cari semua file .txt di direktori saat ini
current_dir = os.path.dirname(os.path.abspath(__file__))
txt_files = glob.glob(os.path.join(current_dir, "*.txt"))

if not txt_files:
    raise FileNotFoundError("Tidak ada file .txt ditemukan.")

print("File .txt ditemukan:")
for i, file in enumerate(txt_files):
    print(f"{i}: {os.path.basename(file)}")

choice = int(input("Pilih nomor file: "))
file_path = txt_files[choice]

# Baca file
df = pd.read_csv(file_path, sep=';', engine='python')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df['Phone timestamp'] = pd.to_datetime(df['Phone timestamp'])

# Ambil sinyal mentah (mean 3 channel)
df['ppg_raw'] = df[['channel 0', 'channel 1', 'channel 2']].mean(axis=1)
ppg = df['ppg_raw']
ppg = (ppg - ppg.mean()) / ppg.std()

# Waktu dan frekuensi sampling
time = (df['Phone timestamp'] - df['Phone timestamp'].iloc[0]).dt.total_seconds()
fs = 1 / np.mean(np.diff(time))

# Filter
ppg_filtered = bandpass_filter(ppg.values, 0.5, 5.0, fs)

# Deteksi high peaks
min_distance = int(fs * 60 / 180)  # 180 bpm maksimal
high_peaks, _ = find_peaks(ppg_filtered, distance=min_distance, prominence=0.5)

# Hitung RR intervals (detik)
rr_intervals = np.diff(time[high_peaks])

# Hitung HRV time-domain features
sdnn = np.std(rr_intervals)
rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
sdsd = np.std(np.diff(rr_intervals))
pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals) * 100

print("\nFitur HRV (Time-Domain):")
print(f"SDNN   : {sdnn:.4f} s")
print(f"RMSSD  : {rmssd:.4f} s")
print(f"SDSD   : {sdsd:.4f} s")
print(f"pNN50  : {pnn50:.2f} %")

# Plot RR Interval
plt.figure(figsize=(10, 4))
plt.plot(rr_intervals, marker='o')
plt.title("RR Interval (NN Interval) dari Sinyal PPG")
plt.xlabel("Beat ke-n")
plt.ylabel("Interval (detik)")
plt.grid(True)
plt.tight_layout()
plt.show()
