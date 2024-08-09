
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

    def run(self, steps=100, save_steps=None):
        for i in range(steps):
            self.run_one_step()
            # show progress
            print('', end='\r')
            print('{:.1f} % finished'.format(i / steps * 100), end='\r')

            if i == steps-1:
                self._plot()
                self.ims.append([self.surf])
                self.y_cuts.append([np.arange(self._xgrid), self.h[:, self.y_cut]])

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

    def compare_fft(self, experimental_comparison_data, filename):
        # Save the plot for the generated surface
        print(filename)
        output_file = os.path.join("Images", filename+'_surface_generated.png')
        plt.title(filename+' Surface Generated')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        # Numerical Data
        profile = self.y_cuts[-1]
        profile_offset = np.mean(profile[1]) 
        profile[1] = profile[1]- profile_offset # Align the profile data with zero on the y-axis

        plt.figure(figsize=(12, 6))
        plt.plot(profile[0], profile[1], label='Numerical Data')
        plt.plot(experimental_comparison_data[0], experimental_comparison_data[1], label='Experimental Data') #Transforms x to value un mm since it is in m
        plt.ylim(-25,25)
        plt.grid(True)  # Add grid if needed
        plt.legend()

        # Compute FFT comparison
        time_values = profile[0]/1000 # Needs to be divided to obtain same as test file
        dt = np.mean(np.diff(time_values))  # Compute the average time step
        # Perform FFT on experimental data
        fft_result_exp = np.fft.fft(profile[1])*dt
        fft_freq_exp = np.fft.fftfreq(len(profile[1]), d=dt)*dt


        # Calculate for experimental data
        # Perform FFT
        time_values = experimental_comparison_data[0]/1000
        dt = np.mean(np.diff(time_values))
        # Perform FFT on experimental data
        fft_result = np.fft.fft(experimental_comparison_data[1])*dt
        fft_freq = np.fft.fftfreq(len(experimental_comparison_data[1]), d=dt)*dt
        fft_exp = np.abs(fft_result)

        # Save the plot
        output_file = os.path.join("Images", filename+'_profile_comparison.png')

        # Adjust layout and save the figure
        plt.title(filename+' Profile Comparison')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        plt.figure(figsize=(6, 6))
        # Subplot 1: Experimental FFT
        plt.subplot(3, 1, 1)
        plt.plot(fft_freq_exp, np.abs(fft_result_exp), color='blue')
        plt.xlim(0,0.015)
        plt.title('Numerical FFT '+filename)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Subplot 2: Experimental FFT with peak highlight
        plt.subplot(3, 1, 2)
        plt.plot(fft_freq, np.abs(fft_result), color='green')
        plt.fill_between(fft_freq[6:25], 0, fft_exp[6:25], color='red', alpha=0.3, label='Peak Region')
        #plt.scatter(fft_freq[peak_index], fft_exp[peak_index], color='red', label='Peak')
        plt.xlim(0,0.015)
        plt.title('Experimental FFT '+filename)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # Subplot 3: Combined
        plt.subplot(3, 1, 3)
        plt.plot(fft_freq_exp, np.abs(fft_result_exp), label='Numerical FFT', color='blue')
        plt.plot(fft_freq, np.abs(fft_result), label='Experimental FFT', linestyle='--', color='green')
        plt.xlim(0,0.015)
        plt.legend()
        plt.title('Combined FFT Comparison '+filename)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Save the plot
        output_file = os.path.join("Images", filename+'_fft_comparison.png')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        plt.show()

    def extract_experimental_fft(self):
        # Numerical Data
        profile = self.y_cuts[-1]
        profile_offset = np.mean(profile[1]) 
        profile[1] = profile[1]- profile_offset # Align the profile data with zero on the y-axis

        # Calculate for numerical data
        # Perform FFT
        time_values = profile[0]/1000 # Needs to be divided to obtain same as test file
        dt = np.mean(np.diff(time_values))  # Compute the average time step
        # Perform FFT on experimental data
        fft_result_exp = np.fft.fft(profile[1])*dt
        fft_freq_exp = np.fft.fftfreq(len(profile[1]), d=dt)*dt

        return [fft_freq_exp, np.abs(fft_result_exp), profile[0],profile[1] ]




if __name__ == "__main__":

    cb = CellBedform(grid=(100, 100))
    cb.run(steps=10)