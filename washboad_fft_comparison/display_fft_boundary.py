import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# CONSTANTS
BOUNDARY_SELECTIONS = [{'velocity': '0.78ms', 'start_index': 6, 'end_index': 25, 'start_value': 0.001348314606741573, 'end_value': 0.0056179775280898875}]

# EXPERIMENTAL DATA FILES MANAGEMENT
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
EXPERIMENTAL_COMPARISON_FILE = "Vuelta80.txt"
SKIPROWS_FILES = 1
D_X = 4450

clicks = []  # To store the indices of clicks
boundaries = {} # To store the boundaries


def load_experimental_data(file_path):
    """Load and preprocess experimental data obtaining its fft and interpolating it to 4450 mm."""
    data = np.loadtxt(file_path, skiprows=SKIPROWS_FILES) # Load file
    offset = np.mean(data[:, 1]) # Center the Signal on the axis
    data[:, 1] -= offset
    data[:,0] = data[:,0] * 1000 - min(data[:,0]) * 1000 # Normalize and transforming to mm
    f = interp1d(data[:,0], data[:,1]) # Interpolation
    array = np.arange(0, D_X, 1)
    y_inter = f(array)
    data_inter = np.array([array, y_inter])
    return data_inter


def main():
    for selection in BOUNDARY_SELECTIONS:
        experimental_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, selection["velocity"], EXPERIMENTAL_COMPARISON_FILE)

        experimental_comparison_data = load_experimental_data(experimental_file_path)

        # Perform FFT
        time_values = experimental_comparison_data[0] / 1000
        dt = np.mean(np.diff(time_values))
        fft_result = np.abs(np.fft.fft(experimental_comparison_data[1]) * dt)
        fft_freq = np.fft.fftfreq(len(experimental_comparison_data[1]), d=dt) * dt

        plt.figure(figsize=(9, 6))
        # Plot FFT
        plt.plot(fft_freq, fft_result, color='green')
        plt.xlim(0, 0.015)
        plt.title(f'Experimental FFT {selection["velocity"]}')
        plt.grid(True)

        plt.fill_between(fft_freq[selection["start_index"]:selection["end_index"]], 0, fft_result[selection["start_index"]:selection["end_index"]], color='red', alpha=0.3, label='Selected Region')
        plt.draw()

        plt.figure(figsize=(9, 6))
        plt.plot(experimental_comparison_data[0], experimental_comparison_data[1], label='Experimental Data') #Transforms x to value un mm since it is in m
        plt.title(f'Profile {selection["velocity"]}')
        plt.ylim(-20,20)
        plt.grid(True)  # Add grid if needed
        plt.legend()

        plt.show()

if __name__ == "__main__":
    main()
