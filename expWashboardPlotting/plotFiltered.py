import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import os

# EXPERIMENTAL DATA FILES MANAGEMENT
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
TEST_FOLDER = "0.78ms"
FILE = "Vuelta80.txt"

# Load the data into a NumPy array
filename = os.path.join("ExperimentalData", CONDITIONS_FOLDER, TEST_FOLDER, FILE)
print(filename)
Data = np.loadtxt(filename, skiprows=1)

# Apply a low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Parameters for the low-pass filter
cutoff = 0.008  # desired cutoff frequency of the filter in Hz
fs = 1.0  # sample rate, in Hz

# Filter the data
filtered_signal = butter_lowpass_filter(Data[:, 1], cutoff, fs)

# Create the filename for the filtered data
filtered_filename = filename.replace(".txt", "_filtered.txt")

# Combine the x-axis and filtered data for saving
filtered_data = np.column_stack((Data[:, 0], filtered_signal))

# Save the filtered data to a new text file
np.savetxt(filtered_filename, filtered_data, delimiter='\t', header='X(mm)\tZ(mm)', comments='')


# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(Data[:, 0], Data[:, 1], label='Original Signal')
plt.plot(Data[:, 0], filtered_signal, label='Filtered Signal')
plt.xlabel('X (m)', fontname='Times New Roman', fontweight='bold')
plt.ylabel('Z (mm)', fontname='Times New Roman', fontweight='bold')
plt.title('Original and Filtered Signal', fontname='Times New Roman', fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
