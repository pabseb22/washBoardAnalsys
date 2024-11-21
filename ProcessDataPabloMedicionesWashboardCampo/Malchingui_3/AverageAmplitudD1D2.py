import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# File paths and x-ranges for analysis
TEST_FOLDERS_D1D2 = ["datos_d1md2_1.txt", "datos_d1md2_2.txt", "datos_d1md2_3.txt", "datos_d1md2_4.txt"]


def obtain_average_amplitude(profile,min_distance ):

    x_values = profile[:, 0]
    y_values = profile[:, 1]

    # Find crests and troughs on the filtered signal
    peaks, _ = find_peaks(y_values, distance=min_distance)
    crests = y_values[peaks]

    troughs, _ = find_peaks(-y_values, distance=min_distance)
    trough_values = y_values[troughs]

    # Calculate average amplitude for this profile
    if len(crests) > 0 and len(trough_values) > 0:
        average_amplitude = (abs(np.mean(crests)) + abs(np.mean(trough_values)))

    # Plot Y-cut profile and identified peaks/troughs
    plt.figure(figsize=(6,6))
    plt.plot(x_values, y_values, label='Original Profile')
    plt.plot(x_values[peaks], crests, "x", label='Peaks')
    plt.plot(x_values[troughs], trough_values, "o", label='Troughs')
    plt.title(f'Y-cut Profile')
    plt.xlabel('Distance (X)')
    plt.ylabel('Elevation')
    plt.legend()
    plt.grid(True)

    print(average_amplitude)

    plt.show()



def main():
    print("Avergae Amplitudes: ")
    for idx, d1d2_file in enumerate(TEST_FOLDERS_D1D2):
        # print(f"\nProcessing files: {d1d2_file}")

        # Load and filter d1d2 data
        data_d1d2 = np.loadtxt(d1d2_file)


        obtain_average_amplitude(data_d1d2, 8000)

if __name__ == "__main__":
    main()
