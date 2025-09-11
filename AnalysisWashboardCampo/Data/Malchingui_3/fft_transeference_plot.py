import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Configuration for File Processing and Filter
CUTOFF_FREQ = 20       # Cutoff frequency for the low-pass filter (Hz)
FILTER_ORDER = 4       # Filter order

# File paths and x-ranges for analysis
TEST_FOLDERS_D1D2 = ["datos_d1md2_1.txt", "datos_d1md2_2.txt", "datos_d1md2_3.txt", "datos_d1md2_4.txt"]
TEST_FOLDERS_PROFILE = ["datos_perfil_1.txt", "datos_perfil_2.txt", "datos_perfil_3.txt", "datos_perfil_4.txt"]
x_ranges = [ (2.5, 40), (2.5, 40), (15, 40), (15, 50)]
x_ranges_d1d2 = [ (2.5, 40), (2.5, 40), (15, 40), (15, 50)]

# Output directories
OUTPUT_FOLDER = "fft_results"
PROFILE_PATH = os.path.join(OUTPUT_FOLDER, "profile")
ACCELEROMETER_PATH = os.path.join(OUTPUT_FOLDER, "accelerometer")
TRANSFER_PATH = os.path.join(OUTPUT_FOLDER, "transfer_function")

# Create output directories
os.makedirs(PROFILE_PATH, exist_ok=True)
os.makedirs(ACCELEROMETER_PATH, exist_ok=True)
os.makedirs(TRANSFER_PATH, exist_ok=True)

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
    if x_range:
        data = data[(data[:, 0] >= x_range[0]) & (data[:, 0] <= x_range[1])]
    return data

def perform_fft(data):
    """Perform FFT and return frequency and amplitude."""
    time_values, signal = data
    dt = np.mean(np.diff(time_values))
    fft_result = np.fft.fft(signal) * dt
    fft_freq = np.fft.fftfreq(len(signal), d=dt)
    return fft_freq, fft_result

def save_fft_results(fft_freq, fft_result, filename, path):
    """Save FFT results to text file and plot as PNG."""
    output_txt_path = os.path.join(path, f"{filename}_fft.txt")
    np.savetxt(output_txt_path, np.column_stack((fft_freq, np.abs(fft_result))), 
               header="Frequency (Hz)\tAmplitude", delimiter="\t")
    print(f"FFT results saved to {output_txt_path}")

    # Plot FFT results and save as PNG
    plt.figure(figsize=(6, 6))
    plt.plot(fft_freq[0:len(fft_freq)//2], np.abs(fft_result[0:len(fft_freq)//2]), color='blue')
    plt.xlim(0, 10)
    plt.title('FFT_'+filename)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    output_png_path = os.path.join(path, f"{filename}_fft.png")
    plt.savefig(output_png_path)
    plt.close()
    print(f"FFT plot saved to {output_png_path}")

def plot_signals(time_values, original_signal, smoothed_signal, filename, path):
    """Plot the original and smoothed signals."""
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, original_signal, label="Original Signal", color='blue', alpha=0.5)
    plt.plot(time_values, smoothed_signal, label="Smoothed Signal", color='red')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Signal Comparison ")
    plt.legend()
    plt.grid(True)
    output_png_path = os.path.join(path, f"{filename}_profile.png")
    plt.savefig(output_png_path)
    plt.close()
    print(f"Profile plot saved to {output_png_path}")

def calculate_transfer_function(fft_profile, fft_d1d2, fft_freq):
    """Calculate transfer function by dividing profile FFT by d1d2 FFT."""
    return fft_d1d2 / fft_profile

def save_transfer_function(transfer_function, fft_freq, filename):
    """Save transfer function results and plot."""
    output_txt_path = os.path.join(TRANSFER_PATH, f"{filename}_transfer_function.txt")
    np.savetxt(output_txt_path, np.column_stack((fft_freq, np.abs(transfer_function))), 
               header="Frequency (Hz)\tTransfer Amplitude", delimiter="\t")
    print(f"Transfer function saved to {output_txt_path}")

    plt.figure(figsize=(6, 6))
    plt.plot(fft_freq[0:len(fft_freq)//2], np.abs(transfer_function[0:len(fft_freq)//2]), color='purple')
    plt.ylim(0,2)
    plt.xlim(0, 10)
    plt.title('Transfer Function')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    output_png_path = os.path.join(TRANSFER_PATH, f"{filename}_transfer_function.png")
    plt.savefig(output_png_path)
    plt.close()
    print(f"Transfer function plot saved to {output_png_path}")

def plot_all_transfer_functions(transfer_functions, fft_freq, filenames):
    """Plot all transfer functions in subplots and save as a single image."""
    n = len(transfer_functions)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n))
    
    for i, (transfer_function, filename) in enumerate(zip(transfer_functions, filenames)):
        ax = axes[i]
        ax.plot(fft_freq[0:len(fft_freq)//2], np.abs(transfer_function[0:len(fft_freq)//2]), color='purple')
        ax.set_xlim(0, 10)
        ax.set_ylim(0,1)
        ax.set_title(f'TF#{filename} / Malchingui 3')
        ax.set_xlabel('1/'+r'$\lambda$'+' (1/m)')
        ax.set_ylabel('TF(-)')
        ax.grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(TRANSFER_PATH, "all_transfer_functions.png")
    plt.savefig(output_path)
    plt.close()
    print(f"All transfer functions saved to {output_path}")

def main():
    transfer_functions = []
    filenames = []

    for idx, (profile_file, d1d2_file) in enumerate(zip(TEST_FOLDERS_PROFILE, TEST_FOLDERS_D1D2)):
        print(f"\nProcessing files: {profile_file} and {d1d2_file}")
        
        # Define x-range for the pair of files
        x_range = x_ranges[idx]
        x_range_d1d2 = x_ranges_d1d2[idx]

        # Load and filter profile data
        data_profile = load_data(profile_file, x_range=x_range)
        time_values_profile, smoothed_profile = low_pass_filter(data_profile)

        # Load and filter d1d2 data
        data_d1d2 = load_data(d1d2_file, x_range=x_range_d1d2)

        # Plot original and smoothed signals for comparison
        plot_signals(time_values_profile, data_profile[:, 1], smoothed_profile, filename=os.path.splitext(profile_file)[0], path=PROFILE_PATH)
        plot_signals(data_d1d2[:, 0], data_d1d2[:, 1], data_d1d2[:, 1], filename=os.path.splitext(d1d2_file)[0], path=ACCELEROMETER_PATH)

        # Perform FFT for profile and d1d2 data
        fft_freq, fft_profile = perform_fft((time_values_profile, smoothed_profile))
        _, fft_d1d2 = perform_fft((data_d1d2[:, 0], data_d1d2[:, 1]))

        # Save FFT results for each signal
        save_fft_results(fft_freq, fft_profile, filename=os.path.splitext(profile_file)[0], path=PROFILE_PATH)
        save_fft_results(fft_freq, fft_d1d2, filename=os.path.splitext(d1d2_file)[0], path=ACCELEROMETER_PATH)

        # Calculate and save transfer function
        transfer_function = calculate_transfer_function(fft_profile, fft_d1d2, fft_freq)
        save_transfer_function(transfer_function, fft_freq, filename=f"transfer_{idx + 1}")
        transfer_functions.append(transfer_function)
        filenames.append(f"transfer_{idx + 1}")

    # Plot and save all transfer functions in one image
    plot_all_transfer_functions(transfer_functions, fft_freq, filenames)

if __name__ == "__main__":
    main()
