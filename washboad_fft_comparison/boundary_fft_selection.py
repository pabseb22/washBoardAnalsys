import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# CONSTANTS
VELOCITY = ["0.78ms","1.03ms","1.29ms","1.55ms","2.08ms","2.61ms","3.15ms"]


# EXPERIMENTAL DATA FILES MANAGEMENT
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
EXPERIMENTAL_COMPARISON_FILE = "Vuelta80.txt"
SKIPROWS_FILES = 1
D_X = 4450

clicks = []  # To store the indices of clicks
boundaries = [] # To store the boundaries


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


def on_click(event, fft_freq, fft_result):
    """Event handler for mouse clicks."""
    if event.inaxes:
        x = event.xdata
        index = (np.abs(fft_freq - x)).argmin()
        clicks.append(index)
        if len(clicks) == 2:
            start_index = min(clicks)
            end_index = max(clicks)
            plt.fill_between(fft_freq[start_index:end_index], 0, fft_result[start_index:end_index], color='red', alpha=0.3, label='Selected Region')
            plt.draw()
            print(f"{SELECTED_VELOCITY} boundaries defined at: {fft_freq[start_index]}, {fft_freq[end_index]}")
            clicks.clear()  # Reset clicks for next selection
            boundaries.append({"velocity": SELECTED_VELOCITY, "start_index": start_index, 
                                             "end_index": end_index, "start_value": fft_freq[start_index], "end_value": fft_freq[end_index]})

def main():
    for velocity in VELOCITY:
        global SELECTED_VELOCITY
        SELECTED_VELOCITY = velocity

        experimental_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, velocity, EXPERIMENTAL_COMPARISON_FILE)

        experimental_comparison_data = load_experimental_data(experimental_file_path)

        # Perform FFT
        time_values = experimental_comparison_data[0] / 1000
        dt = np.mean(np.diff(time_values))
        fft_result = np.fft.fft(experimental_comparison_data[1]) * dt
        fft_freq = np.fft.fftfreq(len(experimental_comparison_data[1]), d=dt) * dt

        fig, ax = plt.subplots(figsize=(6, 9))

        # Plot FFT
        ax.plot(fft_freq, np.abs(fft_result), color='green')
        ax.set_xlim(0, 0.015)
        ax.set_title(f'Experimental FFT {velocity}')
        ax.grid(True)

        # Connect the event handler
        fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, fft_freq, np.abs(fft_result)))
        
        plt.figure(figsize=(9, 6))
        plt.plot(experimental_comparison_data[0], experimental_comparison_data[1], label='Experimental Data') #Transforms x to value un mm since it is in m
        plt.title(f'Profile {velocity}')
        plt.ylim(-20,20)
        plt.grid(True)  # Add grid if needed
        plt.legend()

        plt.show()

    print(boundaries)


if __name__ == "__main__":
    main()
