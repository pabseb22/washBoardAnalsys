
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import os
import pandas as pd
from scipy.interpolate import interp1d

class CellBedform():

    def __init__(self, grid=(100, 50), D=0.8, Q=0.6, L0=7.3, b=2.0, y_cut=10, h=np.random.rand(100, 50)):

        # Copy input parameters
        self._xgrid = grid[0]
        self._ygrid = grid[1]
        self.D = D
        self.Q = Q
        self.L0 = L0
        self.b = b
        self.h = h

        self.L = np.empty(self.h.shape)
        self.dest = np.empty(self.h.shape)
        # Make arrays for indeces showing grid of interest and neighbor grids
        self.y, self.x = np.meshgrid(np.arange(self._ygrid), np.arange(self._xgrid))
        self.xminus = self.x - 1
        self.xplus = self.x + 1
        self.yminus = self.y - 1
        self.yplus = self.y + 1

        # Periodic boundary condition
        self.xminus[0, :] = self._xgrid - 1
        self.xplus[-1, :] = 0
        self.yminus[:, 0] = self._ygrid - 1
        self.yplus[:, -1] = 0

        # Variables for visualization
        self.f = plt.figure(figsize=(8, 8))
        self.ax = self.f.add_subplot(111, projection='3d', azim=120)
        self.surf = None
        self.ims = []
        self.y_cuts = []
        self.y_cut = y_cut
        self.amplitudes = []
        self.wavelengths = []

    def run(self, steps=100, save_steps=None, folder='test'):
        for i in range(steps):
            self.run_one_step()
            # show progress
            print('', end='\r')
            print('{:.1f} % finished'.format(i / steps * 100), end='\r')

            if save_steps is not None and i not in save_steps:
                    continue
            self._plot()
            self.ims.append([self.surf])
            self.y_cuts.append([np.arange(self._xgrid), self.h[:, self.y_cut]])
            profile = [np.arange(self._xgrid), self.h[:, self.y_cut]]
            # # Compute FFT for the Y-cut profiles
            # time_values = profile[0]
            # signal_values = profile[1]
            # dt = np.mean(np.diff(time_values))  # Compute the average time step
            # profile_fft = np.fft.fftshift(np.fft.fft(signal_values))
            # frequencies = np.fft.fftshift(np.fft.fftfreq(len(signal_values), dt))

            # # Find peaks
            # peaks, _ = find_peaks(np.abs(profile_fft), height=0)  # Adjust the height parameter based on your data

            # # Calculate amplitude and wavelength from the FFT result
            # if len(peaks) > 0:
            #     amplitude = np.abs(profile_fft[peaks[0]])
            #     wavelength = 1 / frequencies[peaks[0]]
            # else:
            #     amplitude = 0
            #     wavelength = 0

            # # Perform FFT
            # fft_result = np.fft.fft(signal_values)
            # fft_freq = np.fft.fftfreq(len(signal_values), dt)

            # # Calculate amplitude spectrum
            # amplitude_spectrum = 2*np.abs(fft_result) / len(signal_values)

            # # Remove DC component (frequency at index 0)
            # amplitude_spectrum = amplitude_spectrum[1:]
            # fft_freq = fft_freq[1:]

            # # Find the index of the maximum amplitude
            # max_amplitude_index = np.argmax(amplitude_spectrum)

            # # Extract dominant frequency and amplitude
            # dominant_frequency = fft_freq[max_amplitude_index]
            # dominant_amplitude = amplitude_spectrum[max_amplitude_index]

            # # Calculate wavelength of the dominant frequency
            # dominant_wavelength = 1 / dominant_frequency
            # # Save amplitude and wavelength for each step
            # self.amplitudes.append(dominant_amplitude)
            # self.wavelengths.append(dominant_wavelength)

            # plt.figure()
            # plt.subplot(2, 1, 1)
            # plt.plot(time_values, signal_values)
            # plt.title('Original Signal')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')

            # plt.subplot(2, 1, 2)
            # plt.plot(fft_freq, amplitude_spectrum)
            # plt.scatter(dominant_frequency, dominant_amplitude, color='red', marker='x', label='Dominant Frequency')
            # plt.title('Amplitude Spectrum')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Amplitude')
            # plt.legend()

            # plt.tight_layout()
            # plt.show()

        # show progress
        print('', end='\r')
        print('100.0 % finished')

    def run_one_step(self):
        x = self.x
        y = self.y
        xplus = self.xplus
        yplus = self.yplus
        xminus = self.xminus
        yminus = self.yminus
        D = self.D
        Q = self.Q
        L0 = self.L0
        b = self.b
        L = self.L
        dest = self.dest
        self.h = self.h + D * (-self.h + 1. / 6. *
                               (self.h[xplus, y] + self.h[xminus, y] + self.
                                h[x, yplus] + self.h[x, yminus]) + 1. / 12. *
                               (self.h[xplus, yplus] + self.h[xplus, yminus] +
                                self.h[xminus, yplus] + self.h[xminus, yplus]))

        L = L0 + b * self.h  # Length of saltation
        L[np.where(L < 0)] = 0  # Avoid backward saltation
        np.round(L + x, out=dest)  # Grid number must be integer
        np.mod(dest, self._xgrid, out=dest)  # periodic boundary condition
        self.h = self.h - Q  # Entrainment
        for j in range(self.h.shape[0]):  # Settling
            self.h[dest[j, :].astype(np.int32),
                   y[j, :]] = self.h[dest[j, :].astype(np.int32), y[j, :]] + Q

    def _plot(self):
        self.ax.set_zlim3d(-20, 150)
        self.ax.set_xlabel('Distance (X)')
        self.ax.set_ylabel('Distance (Y)')
        self.ax.set_zlabel('Elevation')
        self.surf = self.ax.plot_surface(
            self.x,
            self.y,
            self.h,
            cmap='jet',
            vmax=5.0,
            vmin=-5.0,
            antialiased=True)


    def save_images(self, folder='test', filename='bed', save_steps=None):
        try:
            if len(self.ims) == 0:
                raise Exception('Run the model before saving images.')

            # Create the main folder if it doesn't exist
            os.makedirs("Results", exist_ok=True)
            folder = os.path.join("Results", folder)

            # Create the main folder if it doesn't exist
            os.makedirs(folder, exist_ok=True)

            # Create a subfolder for Steps
            steps_folder = os.path.join(folder, f'steps_{filename}')
            os.makedirs(steps_folder, exist_ok=True)

            # Create a subfolder for Steps Images
            steps_images_folder = os.path.join(folder, f'steps_{filename}_images')
            os.makedirs(steps_images_folder, exist_ok=True)

            # Create a subfolder for Y-cut profiles
            y_cut_folder = os.path.join(folder, f'{filename}_y={self.y_cut}')
            os.makedirs(y_cut_folder, exist_ok=True) if self.y_cut is not None else None

            # Create a subfolder for Y-cut profile images
            y_cut_images_folder = os.path.join(folder, f'{filename}_y={self.y_cut}_images')
            os.makedirs(y_cut_images_folder, exist_ok=True) if self.y_cut is not None else None

            for i in range(len(save_steps)):
                # show progress
                print('', end='\r')
                print('Saving images... {:.1f}%'.format(i / len(self.ims) * 100), end='\r')

                # Save images to the specified folder
                plt.cla()  # Clear axes
                self.ax.add_collection3d(self.ims[i][0])  # Load surface plot
                self.ax.autoscale_view()
                self.ax.set_zlim3d(-20, 150)
                self.ax.set_xlabel('Distance (X)')
                self.ax.set_ylabel('Distance (Y)')
                self.ax.set_zlabel('Elevation')
                plt.savefig(os.path.join(steps_images_folder, f'{filename}_{i:04d}.png'))
                steps_filename = os.path.join(steps_folder, f'step_{i:04d}.txt')
                elevation_data = self.ims[i][0].get_array()
                np.savetxt(steps_filename, elevation_data)

                # Save Y-cut profiles
                profile = self.y_cuts[i]
                #profile_filename = os.path.join(y_cut_folder, f'step_{i:04d}.txt')
                #np.savetxt(profile_filename, np.column_stack(profile), comments="", delimiter="    ",fmt="%d %.9f")

                df = pd.DataFrame(np.column_stack(profile), columns=['Index', 'Value'])

                # Specify the filename for the Excel file. This uses the same naming convention as before.
                profile_filename = os.path.join(y_cut_folder, f'step_{i:04d}.xlsx')

                # Save the DataFrame to an Excel file. The engine='openpyxl' is specified to ensure compatibility with .xlsx format.
                df.to_excel(profile_filename, index=False, engine='openpyxl')

                # Save Y-cut profile images
                plt.figure()
                plt.plot(profile[0], profile[1])
                plt.title(f'Y-cut Profile at Y={self.y_cut} (Step {i})')
                plt.xlabel('Distance (X)')
                plt.ylabel('Elevation')
                plt.savefig(os.path.join(y_cut_images_folder, f'profile_step_{i:04d}.png'))
                plt.close()

            print('Done. All data were saved and cleared.')

        except Exception as error:
            print('Unexpected error occurred.')
            print(error)

    def compare_fft(self, save_steps=None, folder='test'):
        file_path_exp = os.path.join("ExperimentalData", "80thPass2ms.txt")
        data_exp = np.loadtxt(file_path_exp)
        profile = self.y_cuts[-1]
        # Align the profile data with zero on the y-axis
        profile_offset = np.mean(profile[1])
        profile[1] = profile[1]- profile_offset
        data_exp_offset = np.mean(data_exp[:, 1])
        data_exp[:, 1] = data_exp[:, 1] - data_exp_offset

        plt.figure(figsize=(12, 6))
        plt.plot(profile[0], profile[1], label='Numerical Data')
        plt.plot(data_exp[:, 0]*1000, data_exp[:, 1], label='Experimental Data') #Transforms x to value un mm since it is in m
        plt.grid(True)  # Add grid if needed

        # Compute FFT comparison
        time_values = profile[0]/1000 # Needs to be divided to obtain sabe as test file
        dt = np.mean(np.diff(time_values))  # Compute the average time step
        # Perform FFT on experimental data
        fft_result_exp = np.fft.fft(profile[1])
        fft_freq_exp = np.fft.fftfreq(len(profile[1]), d=dt)


        # Calculate for experimental data
        # Perform FFT
        time_values = data_exp[:, 0]
        dt = np.mean(np.diff(time_values))
        # Perform FFT on experimental data
        fft_result = np.fft.fft(data_exp[:, 1])
        fft_freq = np.fft.fftfreq(len(data_exp[:, 1]), d=dt)


        plt.figure(figsize=(6, 6))

        # Subplot 1: Experimental FFT
        plt.subplot(3, 1, 1)
        plt.plot(fft_freq_exp, np.abs(fft_result_exp), color='blue')
        plt.title('Experimental FFT')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Subplot 2: Comparison FFT
        plt.subplot(3, 1, 2)
        plt.plot(fft_freq, np.abs(fft_result), color='green')
        plt.title('Numerical FFT')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Subplot 3: Combined
        plt.subplot(3, 1, 3)
        plt.plot(fft_freq_exp, np.abs(fft_result_exp), label='Experimental FFT', color='blue')
        plt.plot(fft_freq, np.abs(fft_result), label='Comparison FFT', linestyle='--', color='green')
        plt.title('Combined FFT Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    cb = CellBedform(grid=(100, 100))
    cb.run(steps=10)