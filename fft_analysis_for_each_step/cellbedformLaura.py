"""A component that simulates a bedform formation
 using the cell model of Nishimori and Ouchi (1993)

Reference:
Nishimori, H., & Ouchi, N. (1993). Formation of ripple
 patterns and dunes by wind-blown sand. Physical Review
 Letters, 71(1), 197.

.. codeauthor:: Hajime Naruse

Example
---------------
from cellbedform import CellBedform
cb = CellBedform(grid=(100,100))
cb.run(200)
cb.save_images('bedform')

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.signal import find_peaks
import os


class CellBedform():
    """Cell model for dunes. This model calculate sediment transport on the
       basis of simplified laws. This model assumes that bedload transport
       of sediment particles occurs in two modes: (1) rolling and sliding,
       and (2) saltation.
    """

    def __init__(self, grid=(100, 50), D=0.8, Q=0.6, L0=7.3, b=2.0, y_cut=10):
        """
        Parameters
        -----------------

        grid : list(int, int), optional
            Size of computation grids. Grid sizes of X and Y coordinates
            need to be specified. Default values are 100x100.

        D : float, optional
            Diffusion coefficient for rolling and sliding transport.
            Larger values prescribes larger rates of tranport. Default
            value is 0.8.

        Q : float, optional
            Entrainment rate of saltation transport. Larger values prescribes
            the larger rate of sediment pick-up by flows. Default value is 0.6.

        L0 : float, optional
            Minimum length of saltation transport length. Default is 7.3.

        b : float, optional
            A coefficient to determine saltation length. Larger value
            prescribes longer transport length Default is 2.0.
        """

        # Copy input parameters
        self._xgrid = grid[0]
        self._ygrid = grid[1]
        self.D = D
        self.Q = Q
        self.L0 = L0
        self.b = b

        # Make initial topography
        self.h = np.random.rand(self._xgrid, self._ygrid)
        self.L = np.empty(self.h.shape)
        self.dest = np.empty(self.h.shape)

        # Make arrays for indeces showing grid of interest and neighbor grids
        self.y, self.x = np.meshgrid(
            np.arange(self._ygrid), np.arange(self._xgrid))
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

    def run(self, steps=100):
        """Run the model for specified steps.

        Parameters
        ----------------------

        steps: integer, optional
            Number of steps for calculation. Default value is 100.

        """

        for i in range(steps):
            self.run_one_step()

            # show progress
            print('', end='\r')
            print('{:.1f} % finished'.format(i / steps * 100), end='\r')

            # store animation frames
            self._plot()
            self.ims.append([self.surf])
            self.y_cuts.append([np.arange(self._xgrid), self.h[:, self.y_cut]])



        # show progress
        print('', end='\r')
        print('100.0 % finished')

    def run_one_step(self):
        """Calculate one step of the model
        """

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

        # Rolling and sliding
        # Amounts of sediment transported to/from adjacent grids are 1/6 D h,
        # and those to/from diagonal grids are 1/12 D h
        self.h = self.h + D * (-self.h + 1. / 6. *
                               (self.h[xplus, y] + self.h[xminus, y] + self.
                                h[x, yplus] + self.h[x, yminus]) + 1. / 12. *
                               (self.h[xplus, yplus] + self.h[xplus, yminus] +
                                self.h[xminus, yplus] + self.h[xminus, yplus]))

        # Saltation
        # Length of saltation is determined as:
        # L = L0 + b h
        # Thus, particles at higher elevation travel longer
        L = L0 + b * self.h  # Length of saltation
        L[np.where(L < 0)] = 0  # Avoid backward saltation
        np.round(L + x, out=dest)  # Grid number must be integer
        np.mod(dest, self._xgrid, out=dest)  # periodic boundary condition
        self.h = self.h - Q  # Entrainment
        for j in range(self.h.shape[0]):  # Settling
            self.h[dest[j, :].astype(np.int32),
                   y[j, :]] = self.h[dest[j, :].astype(np.int32), y[j, :]] + Q

    def _plot(self):
        """plot results on the figure.

        This is only to use inside of this module.

        """
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

    def show(self):
        """Show a figure to illustrate result of calculation
        """
        self._plot()
        plt.show()

    def save_images(self, folder='results', filename='bed'):
        """Save image sequence, Y-cut profiles, and Y-cut profile images

        Parameters
        --------------
        folder : str, optional
            The folder where the image sequence, profiles, and profile images will be saved. Default is 'results'.
        filename : str, optional
            File header of the image sequence.
        y_cut : int or None, optional
            The Y-coordinate at which to analyze the cut. If None, no Y-cut profiles will be saved.

        """

        try:
            if len(self.ims) == 0:
                raise Exception('Run the model before saving images.')

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

            for i in range(len(self.ims)):
                # show progress
                print('', end='\r')
                print(
                    'Saving an image sequence...{:.1f}%'.format(
                        i / len(self.ims) * 100),
                    end='\r')

                # Save images to the specified folder
                plt.cla()  # Clear axes
                self.ax.add_collection3d(self.ims[i][0])  # Load surface plot
                self.ax.autoscale_view()
                self.ax.set_zlim3d(-20, 150)
                self.ax.set_xlabel('Distance (X)')
                self.ax.set_ylabel('Distance (Y)')
                self.ax.set_zlabel('Elevation')
                plt.savefig(os.path.join(steps_images_folder, filename + '{:04d}.png'.format(i)))
                steps_filename = os.path.join(steps_folder, f'step_{i:04d}.txt')
                elevation_data = self.ims[i][0].get_array()
                np.savetxt(steps_filename, elevation_data)

                # Save Y-cut profiles
                profile = self.y_cuts[i]
                profile_filename = os.path.join(folder, f'{filename}_y={self.y_cut}', f'step_{i:04d}.txt')
                np.savetxt(profile_filename, np.column_stack(profile), comments="", delimiter=" ")


                # Compute FFT for the Y-cut profiles
                time_values = profile[0]
                signal_values = profile[1]
                dt = np.mean(np.diff(time_values))  # Compute the average time step
                profile_fft = np.fft.fftshift(np.fft.fft(signal_values))
                frequencies = np.fft.fftshift(np.fft.fftfreq(len(signal_values), dt))

                # Save FFT data for Y-cut profiles
                fft_data = np.column_stack((frequencies, np.abs(profile_fft)))

                # Create a subfolder for FFT files
                fft_folder = os.path.join(folder, f'fft_files')
                os.makedirs(fft_folder, exist_ok=True)

                # Save FFT data for Y-cut profiles in the FFT folder
                fft_filename = os.path.join(fft_folder, f'fft_{filename}_y={self.y_cut}_step_{i:04d}.txt')
                np.savetxt(fft_filename, fft_data, comments="", delimiter=" ")

                # Create a subfolder for FFT images
                fft_images_folder = os.path.join(folder, 'fft_files_images')
                os.makedirs(fft_images_folder, exist_ok=True)

                # Plot and save Y-cut profile images
                plt.figure()
                plt.plot(frequencies, np.abs(profile_fft))
                plt.title(f'fft_bed_Y={self.y_cut} (Step {i})')
                plt.xlabel('Frequencies (X)')
                plt.ylabel('FFT')

                # Find peaks
                peaks, _ = find_peaks(np.abs(profile_fft), height=0)  # Adjust the height parameter based on your data

                # Exclude the peak at (0, 0) and filter positive peaks
                positive_peaks = np.array([peak for peak in peaks if frequencies[peak] > 0])

                # Get x and y values of the three highest positive peaks
                sorted_peaks = positive_peaks[np.argsort(np.abs(profile_fft[positive_peaks]))[::-1][:3]]

                peak_frequencies = frequencies[sorted_peaks]
                peak_values = np.abs(profile_fft[sorted_peaks])

                # Show the three highest peaks on the plot
                plt.plot(peak_frequencies, peak_values, 'ro', label='Top 3 Peaks')
                plt.legend()

                # Save the plot image to the FFT images folder
                fft_image_filename = os.path.join(fft_images_folder, f'fft_step_{i:04d}.png')
                plt.savefig(fft_image_filename)
                plt.close()

                # Save x and y values of the three highest peaks into a txt file
                peaks_folder = os.path.join(folder, 'fft_peaks')
                os.makedirs(peaks_folder, exist_ok=True)
                peaks_filename = os.path.join(peaks_folder, f'peaks_step_{i:04d}.txt')
                np.savetxt(peaks_filename, np.column_stack((peak_frequencies, peak_values)), comments="", delimiter=" ")


                # Save Y-cut profile images
                plt.figure()
                plt.plot(profile[0], profile[1])
                plt.title(f'Y-cut Profile at Y={self.y_cut} (Step {i})')
                plt.xlabel('Distance (X)')
                plt.ylabel('Elevation')
                plt.savefig(os.path.join(y_cut_images_folder, f'profile_step_{i:04d}.png'))
                plt.close()


            self.ims = []
            self.y_cuts = []
            print('done. All data were saved and cleared.')

        except Exception as error:
            print('Unexpected error occurred.')
            print(error)

    def animation(self, filename='anim.mp4', format='mp4'):
        """Show and save an animation to exhibit results of simulation.

        Parameters
        -----------------------
        filename: Str, optional
            A file name to store the animation movie.
            The default file name is 'anim.mp4'.

        format: Str, optional
            File format of the movie file. The format 'mp4' or 'gif' can be
            chosen. The softwares ffmpeg and imagemagick are required
            respectively.
        """
        try:
            if self.ims == []:
                raise Exception('Run the model before saving data.')

            ani = animation.ArtistAnimation(self.f, self.ims, interval=100)
            print("Saving a movie file. It may take several 10s minutes...")

            # Check the file format
            if format == 'mp4':
                ani.save(filename, writer="ffmpeg")
            elif format == 'gif':
                ani.save(filename, writer="imagemagick")
            else:
                raise ValueError('Format not supported')

            print("A movie file was created.")

        except ValueError:
            print('Movie format is not supported in this environment. \
                Install ffmpeg or imagemagick for mp4 or gif formats.')
        except Exception:
            print(Exception)


if __name__ == "__main__":

    cb = CellBedform(grid=(100, 100))
    cb.run(steps=10)
    # cb.animation(filename='anim.gif', format='gif')
    # cb.show()
    cb.save_images()
