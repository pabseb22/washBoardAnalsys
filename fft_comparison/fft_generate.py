import os
import matplotlib.pyplot as plt
import numpy as np

# Load experimental data
file_path = os.path.join("ExperimentalData", "80thPass2ms.txt")
data_exp = np.loadtxt(file_path)
data_exp_offset = np.mean(data_exp[:, 1])
data_exp[:, 1] = (data_exp[:, 1] - data_exp_offset)
distance_values = data_exp[:, 0] #Needs to be transforming its data to mm from m
dt = np.mean(np.diff(distance_values))  # Compute the average time step

# Perform FFT on experimental data
fft_result = np.fft.fft(data_exp[:, 1])*dt
fft_freq = np.fft.fftfreq(len(data_exp[:, 1]), d=dt)*dt

# Perform inverse FFT to obtain original data
original_data = np.fft.ifft(fft_result/dt)

# Plot the original data, its FFT, and the reconstructed data
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(distance_values*1000, data_exp[:, 1], color='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(fft_freq, np.abs(fft_result), color='green', label='FFT')
plt.xlim(0,0.005)
plt.title('FFT')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(distance_values*1000, original_data.real, color='red', label='Reconstructed Data')
plt.title('Reconstructed Data from Inverse FFT')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
