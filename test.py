import numpy as np
import matplotlib.pyplot as plt

# Generate a sample signal
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector
f1, f2 = 50, 150  # Frequencies of two sine waves
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Perform FFT
fft_result = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(signal), 1/fs)

# Calculate amplitude spectrum
amplitude_spectrum = np.abs(fft_result) / len(signal)

# Remove DC component (frequency at index 0)
amplitude_spectrum = amplitude_spectrum[1:]
fft_freq = fft_freq[1:]

# Find the index of the maximum amplitude
max_amplitude_index = np.argmax(amplitude_spectrum)

# Extract dominant frequency and amplitude
dominant_frequency = fft_freq[max_amplitude_index]
dominant_amplitude = amplitude_spectrum[max_amplitude_index]

# Calculate wavelength of the dominant frequency
dominant_wavelength = 1 / dominant_frequency

# Plot results
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(fft_freq, amplitude_spectrum)
plt.scatter(dominant_frequency, dominant_amplitude, color='red', marker='x', label='Dominant Frequency')
plt.title('Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

# Display results
print(f"Dominant Frequency: {dominant_frequency} Hz")
print(f"Dominant Amplitude: {dominant_amplitude}")
print(f"Dominant Wavelength: {dominant_wavelength} s")
