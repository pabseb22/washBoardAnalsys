import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# CONSTANTS
BOUNDARY_SELECTIONS = [{'velocity': '0.78ms', 'start_index': np.int64(6), 'end_index': np.int64(25), 'start_value': np.float64(0.001348314606741573), 'end_value': np.float64(0.0056179775280898875)}, 
                       {'velocity': '1.03ms', 'start_index': np.int64(7), 'end_index': np.int64(37), 'start_value': np.float64(0.0015730337078651683), 'end_value': np.float64(0.008314606741573034)}, 
                       {'velocity': '1.29ms', 'start_index': np.int64(14), 'end_index': np.int64(36), 'start_value': np.float64(0.0031460674157303367), 'end_value': np.float64(0.008089887640449439)}, 
                       {'velocity': '1.55ms', 'start_index': np.int64(11), 'end_index': np.int64(33), 'start_value': np.float64(0.0024719101123595504), 'end_value': np.float64(0.007415730337078651)}, 
                       {'velocity': '2.08ms', 'start_index': np.int64(4), 'end_index': np.int64(32), 'start_value': np.float64(0.0008988764044943821), 'end_value': np.float64(0.0071910112359550565)}, 
                       {'velocity': '2.61ms', 'start_index': np.int64(4), 'end_index': np.int64(29), 'start_value': np.float64(0.0008988764044943821), 'end_value': np.float64(0.00651685393258427)}, 
                       {'velocity': '3.15ms', 'start_index': np.int64(3), 'end_index': np.int64(27), 'start_value': np.float64(0.0006741573033707865), 'end_value': np.float64(0.006067415730337079)}]

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
