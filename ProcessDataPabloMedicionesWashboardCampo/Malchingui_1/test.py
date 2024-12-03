import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

# Simulated Signal
fs = 1000  # Sampling frequency (Hz)
T = 1.0  # Duration (seconds)
N = int(fs * T)  # Total samples
t = np.linspace(0, T, N, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.normal(size=t.shape)

# FFT
fft_vals = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, 1/fs)
positive_freqs = freqs[:N//2]
fft_magnitude = np.abs(fft_vals[:N//2]) ** 2  # Power Spectrum

# Normalize to obtain PSD
psd_fft = fft_magnitude / (fs * N)

# Compute PSD with Welch's Method
frequencies, psd_welch = welch(signal, fs, nperseg=256)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, psd_fft, label="PSD from FFT")
plt.semilogy(frequencies, psd_welch, label="PSD from Welch", linestyle="--")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.legend()
plt.grid()
plt.show()
