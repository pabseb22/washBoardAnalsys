
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt, find_peaks
#import pywt

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
        self.scalogram_y_data = []


    def run(self, steps=100, save_steps=None):
        profiles = []
        for i in range(steps):
            self.run_one_step()
            # show progress
            print('', end='\r')
            print('{:.1f} % finished'.format(i / steps * 100), end='\r')

            if i == steps-1:
                self._plot()
                self.ims.append([self.surf])
            self.y_cuts.append([np.arange(self._xgrid), self.h[:, self.y_cut]])

            profiles.append(self.h[:, self.y_cut])

        self.scalogram_y_data = np.hstack(profiles)

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


    def compare_fft(self, experimental_comparison_data, filename,boundaries,control_steps, save_images):
        plt.title(filename+' Surface Generated')

        # Numerical Data
        profile = self.y_cuts[-1]
        profile_offset = np.mean(profile[1]) 
        profile[1] = profile[1]- profile_offset # Align the profile data with zero on the y-axis

        plt.figure(figsize=(12, 6))
        plt.plot(profile[0], profile[1], label='Numerical Data')
        plt.plot(experimental_comparison_data[0], experimental_comparison_data[1], label='Experimental Data') #Transforms x to value un mm since it is in m
        plt.ylim(-50,50)
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

        # Adjust layout and save the figure
        plt.title(filename+' Profile Comparison')
        plt.tight_layout()

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
        plt.fill_between(fft_freq[boundaries[0]:boundaries[1]], 0, fft_exp[boundaries[0]:boundaries[1]], color='red', alpha=0.3, label='Peak Region')
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
        output_file = os.path.join(filename+'_fft_comparison.png')

        # Adjust layout and save the figure
        plt.tight_layout()

        # save Experimental FFT
        # output_file_data_fft = os.path.join(filename,'Exp_fft_80th.xlsx')
        # df = pd.DataFrame({
        #     'Frequency': fft_freq,
        #     'Amplitude': np.abs(fft_result)
        # })
        # df.to_excel(output_file_data_fft, index=False)

        if(save_images):
            output_file_data_fft = os.path.join(filename,'fft_80th.xlsx')
            df = pd.DataFrame({
                'Frequency': fft_freq_exp,
                'Amplitude': np.abs(fft_result_exp)
            })
            df.to_excel(output_file_data_fft, index=False)

            for i in range(len(self.y_cuts)):
                # Analyze Y-cut profiles
                profile = self.y_cuts[i]
                profile_offset = np.mean(profile[1]) 
                profile[1] = profile[1]- profile_offset
                if((i+1) in control_steps):
                    output_file_data = os.path.join(filename,'profile_'+str(i+6)+'th.txt')
                    np.savetxt(output_file_data, profile[1], fmt='%.4f', delimiter='\n')

        # plt.show()

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

    def obtain_average_amplitude(self,min_distance,low_pass,control_steps,filename, save_images):
        amplitudes = []
        for i in range(len(self.y_cuts)):
            # Analyze Y-cut profiles
            profile = self.y_cuts[i]
            x_values = profile[0]
            y_values = profile[1]

            # Apply low-pass filter to smooth the signal
            cutoff_frequency = low_pass  # Adjust this value based on your data
            sampling_rate = 1 / np.mean(np.diff(x_values))  # Assuming uniform spacing
            filtered_y_values = butter_lowpass_filter(y_values, cutoff_frequency, sampling_rate)

            # Find crests and troughs on the filtered signal
            peaks, _ = find_peaks(filtered_y_values, distance=min_distance)
            crests = filtered_y_values[peaks]

            troughs, _ = find_peaks(-filtered_y_values, distance=min_distance)
            trough_values = filtered_y_values[troughs]

            # Calculate average amplitude for this profile
            if len(crests) > 0 and len(trough_values) > 0:
                average_amplitude = (abs(np.mean(crests)) + abs(np.mean(trough_values)))
                amplitudes.append(average_amplitude)

            # Plot Y-cut profile and identified peaks/troughs
            if((i+1) in control_steps):
                plt.figure(figsize=(6,6))
                plt.plot(x_values, y_values, label='Original Profile')
                plt.plot(x_values, filtered_y_values, label='Filtered Profile')
                plt.plot(x_values[peaks], crests, "x", label='Peaks')
                plt.plot(x_values[troughs], trough_values, "o", label='Troughs')
                plt.title(f'Y-cut Profile at Y={self.y_cut} (Step {i+1})')
                plt.xlabel('Distance (X)')
                plt.ylabel('Elevation')
                plt.ylim([-20,20])
                plt.legend()
                plt.grid(True)

        # Plot amplitude development over steps
        plt.figure(figsize=(6,6))
        plt.scatter(range(len(amplitudes)), amplitudes, marker='o', color='b')
        plt.title('Amplitude Development Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Amplitude (mm)')
        plt.grid(True)

        # Save the plot
        output_file = os.path.join(filename,'amplitud_development.png')
        output_file_data = os.path.join(filename,'amplitud_development.txt')

        # Adjust layout and save the figure
        if(save_images):
            np.savetxt(output_file_data, amplitudes, fmt='%.8f', delimiter='\n')


        plt.show()

        print('Done. All data processed.')


    def plot_scalogram(self,filename,velocity, save_images):
        Y = self.scalogram_y_data

        X = np.arange(1, len(Y)+1)

        # Define the parameters for the cmor wavelet
        wavelet = 'cmor1.0-0.5'  # Example parameters; adjust as needed
        scales = np.arange(1, 20)

        # Perform CWT
        coef, freqs = pywt.cwt(Y, scales, wavelet)
        plt.cla()
        # Create the scalogram
        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(coef), aspect='auto', extent=[1, len(Y), 1, 10], cmap='jet', interpolation='bilinear', vmin=0, vmax=20)
        plt.colorbar(label='Magnitude')
        plt.title('Scalogram '+velocity)
        plt.xlabel('Position(mm)')
        plt.ylabel('Frequenzy(Hz)')
        if(save_images):
            output_file = os.path.join(filename,'scalogram.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f'Scalogram saved to {output_file}')

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y



if __name__ == "__main__":

    cb = CellBedform(grid=(100, 100))
    cb.run(steps=10)