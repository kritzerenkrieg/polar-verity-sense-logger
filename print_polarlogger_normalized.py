import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os
import glob

# Fungsi filter bandpass Butterworth
def bandpass_filter(signal, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Cari semua file .txt di direktori script ini
current_dir = os.path.dirname(os.path.abspath(__file__))
txt_files = glob.glob(os.path.join(current_dir, "*.txt"))

if not txt_files:
    raise FileNotFoundError("Tidak ditemukan file .txt di direktori saat ini.")

print("Daftar file .txt yang ditemukan:")
for i, file in enumerate(txt_files):
    print(f"{i}: {os.path.basename(file)}")

choice = int(input("Masukkan nomor file yang akan diproses: "))
if choice < 0 or choice >= len(txt_files):
    raise IndexError("Pilihan file tidak valid.")

file_path = txt_files[choice]

# Baca data dari file
df = pd.read_csv(file_path, sep=';', engine='python')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Hilangkan kolom unnamed
df['Phone timestamp'] = pd.to_datetime(df['Phone timestamp'])

# Ambil rata-rata dari channel 0, 1, 2 sebagai sinyal PPG mentah
df['ppg_raw'] = df[['channel 0', 'channel 1', 'channel 2']].mean(axis=1)

# Normalisasi sinyal: mean-center dan skala deviasi standar
ppg = df['ppg_raw']
ppg_centered = ppg - ppg.mean()
ppg_scaled = ppg_centered / ppg_centered.std()

# Hitung frekuensi sampling berdasarkan waktu timestamp (Hz)
time = (df['Phone timestamp'] - df['Phone timestamp'].iloc[0]).dt.total_seconds()
fs = 1 / np.mean(np.diff(time))

print(f"Estimasi frekuensi sampling (Hz): {fs:.2f}")

# Terapkan filter bandpass 0.5 - 5 Hz untuk isolasi detak jantung manusia
lowcut = 0.5
highcut = 5.0
ppg_filtered = bandpass_filter(ppg_scaled.values, lowcut, highcut, fs, order=3)

# Deteksi puncak (High peaks) dan lembah (Low peaks)
# Set distance minimal antar puncak berdasarkan batas maksimal detak jantung (misal 180 bpm = 3 Hz)
max_hr_bpm = 180
min_distance_samples = int(fs * 60 / max_hr_bpm)  # jarak minimal antar puncak dalam sampel

# Gunakan prominence untuk mengabaikan puncak kecil akibat noise
high_peaks, _ = find_peaks(ppg_filtered, distance=min_distance_samples, prominence=0.5)
low_peaks, _ = find_peaks(-ppg_filtered, distance=min_distance_samples, prominence=0.5)

# Plot hasil
plt.figure(figsize=(14, 6))
plt.plot(time, ppg_filtered, label='Sinyal PPG Terfilter', color='blue', linewidth=1)
plt.xlabel("Waktu (detik)")
plt.ylabel("Sinyal PPG (Ternormalisasi dan Terfilter)")
plt.title("Sinyal PPG dengan Deteksi Puncak Tinggi dan Rendah")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
