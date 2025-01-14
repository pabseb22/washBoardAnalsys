
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class CellBedform():

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

    def run(self, steps=100, save_steps=None):
        for i in range(steps):
            self.run_one_step()

            # show progress
            print('', end='\r')
            print('{:.1f} % finished'.format(i / steps * 100), end='\r')

            if save_steps is not None and i not in save_steps:
                    continue
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

    def save_images(self, folder='test', filename='bed', save_steps=None):
        """
        Save image sequence, Y-cut profiles, and Y-cut profile images at specific steps.

        Parameters
        --------------
        folder : str, optional
            The folder where the image sequence, profiles, and profile images will be saved. Default is 'results'.
        filename : str, optional
            File header of the image sequence.
        save_steps : list of int or None, optional
            List of specific steps at which to save the data. If None, save all steps.

        """

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

            # for i in range(len(self.ims)):
                # if save_steps is not None and i not in save_steps:
                #     continue  # Skip saving if the step is not in the specified list
            for i in range(len(save_steps)):
                # show progress
                print('', end='\r')
                print('Saving images... {:.1f}%'.format(i / len(self.ims) * 100), end='\r')

                # Save images to the specified folder
                plt.cla()  # Clear axes
                self.ax.add_collection3d(self.ims[i][0])  # Load surface plot
                self.ax.autoscale_view()
                self.ax.grid(False)  # Disable the grid
                self.ax.set_zlim3d(-20, 80)
                self.ax.set_xlabel('(X)')
                self.ax.set_ylabel('(Y)')
                self.ax.set_zlabel('(Z)')
                plt.savefig(
                    os.path.join(steps_images_folder, f'{filename}_{i:04d}.png'),
                    dpi=600
                )
                steps_filename = os.path.join(steps_folder, f'step_{i:04d}.txt')
                elevation_data = self.ims[i][0].get_array()
                np.savetxt(steps_filename, elevation_data)

                # Save Y-cut profiles
                profile = self.y_cuts[i]
                profile_filename = os.path.join(y_cut_folder, f'step_{i:04d}.txt')
                np.savetxt(profile_filename, np.column_stack(profile), comments="", delimiter=" ")

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
            print('Done. All data were saved and cleared.')

        except Exception as error:
            print('Unexpected error occurred.')
            print(error)



if __name__ == "__main__":

    cb = CellBedform(grid=(100, 100))
    cb.run(steps=10)
    cb.save_images()
