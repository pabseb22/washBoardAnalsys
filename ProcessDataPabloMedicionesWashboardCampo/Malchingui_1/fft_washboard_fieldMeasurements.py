import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Configuration for File Processing and Filter
TEST_PROFILE = True
CUTOFF_FREQ = 10.0       # Cutoff frequency for the low-pass filter (Hz)
FILTER_ORDER = 4        # Filter order

# File paths and x-ranges for analysis
TEST_FOLDERS_D1D2 = ["datos_d1md2_1.txt"]
# TEST_FOLDERS_D1D2 = ["datos_d1md2_1.txt", "datos_d1md2_2.txt", "datos_d1md2_3.txt", "datos_d1md2_4.txt"]
TEST_FOLDERS_PROFILE = ["datos_perfil_1.txt"]
# TEST_FOLDERS_PROFILE = ["datos_perfil_1.txt", "datos_perfil_2.txt", "datos_perfil_3.txt", "datos_perfil_4.txt"]

x_ranges = [(0.5, 20), (0.5, 30), (0.5, 25), (60, 80)]  # Define x-ranges for each file

# Output directories
OUTPUT_FOLDER = "fft_results"
PROFILE_PATH = os.path.join(OUTPUT_FOLDER, "profile")
ACCELEROMETER_PATH = os.path.join(OUTPUT_FOLDER, "accelerometer")

# Determine which output folder to use
os.makedirs(PROFILE_PATH, exist_ok=True)
os.makedirs(ACCELEROMETER_PATH, exist_ok=True)
OUTPUT_PATH = PROFILE_PATH if TEST_PROFILE else ACCELEROMETER_PATH

def low_pass_filter(data, cutoff=CUTOFF_FREQ, order=FILTER_ORDER):
    """Apply a low-pass Butterworth filter to smooth the signal."""
    time_values, y_values = data[:, 0], data[:, 1]
    dt = np.mean(np.diff(time_values))
    fs = 1 / dt  # Sampling frequency

    # Design the filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter
    smoothed_signal = filtfilt(b, a, y_values)
    return time_values, smoothed_signal

def load_data(file_path, x_range=None):
    """Load and filter data within specified x-range."""
    data = np.loadtxt(file_path)
    if x_range and TEST_PROFILE:
        data = data[(data[:, 0] >= x_range[0]) & (data[:, 0] <= x_range[1])]
    return data

def perform_fft(data, filename):
    """Perform FFT and save the results as PNG and TXT."""
    time_values, signal = data
    dt = np.mean(np.diff(time_values))
    fft_result = np.fft.fft(signal) * dt
    fft_freq = np.fft.fftfreq(len(signal), d=dt)

    # Save FFT results to text file
    output_txt_path = os.path.join(OUTPUT_PATH, f"{filename}_fft.txt")
    np.savetxt(output_txt_path, np.column_stack((fft_freq, np.abs(fft_result))), 
               header="Frequency (Hz)\tAmplitude", delimiter="\t")
    print(f"FFT results saved to {output_txt_path}")

    # Plot FFT results and save as PNG
    plt.figure(figsize=(6, 6))
    plt.plot(fft_freq, np.abs(fft_result), color='blue')
    plt.xlim(0, 3)
    plt.title('FFT')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    output_png_path = os.path.join(OUTPUT_PATH, f"{filename}_fft.png")
    plt.savefig(output_png_path)
    plt.close()
    print(f"FFT plot saved to {output_png_path}")

def plot_signals(time_values, original_signal, smoothed_signal):
    """Plot the original and smoothed signals."""
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, original_signal, label="Original Signal", color='blue', alpha=0.5)
    if TEST_PROFILE:
        plt.plot(time_values, smoothed_signal, label="Smoothed Signal", color='red')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Signal Comparison")
    plt.legend()
    plt.grid(True)


def main():
    files = TEST_FOLDERS_PROFILE if TEST_PROFILE else TEST_FOLDERS_D1D2
    for idx, file_name in enumerate(files):
        print(f"\nProcessing file: {file_name}")
        experimental_file_path = os.path.join(file_name)
        x_range = x_ranges[idx]

        # Load data and apply x-range filter
        data_exp = load_data(experimental_file_path, x_range=x_range)

        # Apply low-pass filter
        time_values, smoothed_signal = low_pass_filter(data_exp)

        # Plot original and smoothed signals for comparison
        plot_signals(time_values, data_exp[:, 1], smoothed_signal)

        # Perform FFT and save the results
        perform_fft((time_values, smoothed_signal), filename=os.path.splitext(file_name)[0])

if __name__ == "__main__":
    main()
