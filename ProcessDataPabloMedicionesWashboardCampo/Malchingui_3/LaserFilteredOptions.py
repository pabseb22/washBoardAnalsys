import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


CUTOFF_FREQ = 0.8     # Cutoff frequency for the low-pass filter (Hz) high: 0.8  low: 5
FILTER_ORDER = 4        # Filter order
FILTER_TYPE = 'high'

# File paths and x-ranges for analysis
# TEST_FOLDERS_LASER = ["datos_perfil_4.txt"]
TEST_FOLDERS_LASER = ["datos_perfil_1.txt", "datos_perfil_2.txt", "datos_perfil_3.txt", "datos_perfil_4.txt"]

# Output directories
OUTPUT_FOLDER_HIGH = "laser_Filtered_highPass"
OUTPUT_FOLDER_LOW = "laser_Filtered_lowPass"
OUTPUT_FOLDER = OUTPUT_FOLDER_HIGH if FILTER_TYPE == 'high' else OUTPUT_FOLDER_LOW


# Determine which output folder to use
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def low_pass_filter(data, cutoff=CUTOFF_FREQ, order=FILTER_ORDER):
    """Apply a low-pass Butterworth filter to smooth the signal."""
    time_values, y_values = data[:, 0], data[:, 1]
    dt = np.mean(np.diff(time_values))
    fs = 1 / dt  # Sampling frequency

    # Design the filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=FILTER_TYPE)

    # Apply the filter
    smoothed_signal = filtfilt(b, a, y_values)
    return time_values, smoothed_signal


def perform_fft(data, filename):
    """Perform FFT and save the results as PNG and TXT."""
    time_values, signal = data
    dt = np.mean(np.diff(time_values))
    fft_result = np.fft.fft(signal) * dt
    fft_freq = np.fft.fftfreq(len(signal), d=dt)

    # Save FFT results to text file
    output_txt_path = os.path.join(f"filtered_{filename}")
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
    output_png_path = os.path.join(f"{filename}_fft.png")
    plt.savefig(output_png_path)
    plt.close()
    print(f"FFT plot saved to {output_png_path}")

def plot_signals(time_values, original_signal, smoothed_signal,filename):
    """Plot the original and smoothed signals."""
    output_txt_path = os.path.join(OUTPUT_FOLDER,f"filtered_{filename}")
    np.savetxt(output_txt_path, np.column_stack((time_values, smoothed_signal)), 
            header="X \tZ", delimiter="\t")
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, original_signal, label="Original Signal", color='blue')
    plt.plot(time_values, smoothed_signal, label="Smoothed Signal", color='red')
    # plt.ylim(-40, 40)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Signal Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    for idx, file_name in enumerate(TEST_FOLDERS_LASER):
        print(f"\nProcessing file: {file_name}")
        experimental_file_path = os.path.join(file_name)

        # Load data and apply x-range filter
        data_exp =  np.loadtxt(experimental_file_path)

        # Apply low-pass filter
        time_values, smoothed_signal = low_pass_filter(data_exp)

        # Plot original and smoothed signals for comparison
        plot_signals(time_values, data_exp[:, 1], smoothed_signal, file_name)

        # Perform FFT and save the results
        #perform_fft((time_values, smoothed_signal), filename=os.path.splitext(file_name)[0])

if __name__ == "__main__":
    main()
