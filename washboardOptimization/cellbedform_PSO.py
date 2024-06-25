
import numpy as np
import matplotlib.pyplot as plt

class CellBedform():

    def __init__(self, grid=(100, 50), D=0.8, Q=0.6, L0=7.3, b=2.0, y_cut=10, h=np.random.rand(100, 50)):
        # Copy input parameters
        self._xgrid = grid[0]
        self._ygrid = grid[1]
        self.D = D
        self.Q = Q
        self.L0 = L0
        self.b = b

        # Make initial topography
        self.h = h
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
        self.y_cuts = []
        self.y_cut = y_cut

    def run(self, steps=100):
        for i in range(steps):
            self.run_one_step()
            self.y_cuts.append([np.arange(self._xgrid), self.h[:, self.y_cut]])

        profile = self.y_cuts[-1]
        profile_offset = np.mean(profile[1])
        profile[1] = profile[1]- profile_offset

        #Compare results
        # Compute FFT comparison
        position_values = profile[0]/1000
        dt = np.mean(np.diff(position_values))  # Compute the average time step
        
        # Perform FFT on profile data
        fft_result = np.fft.fft(profile[1]) * dt
        fft_freq = np.fft.fftfreq(len(profile[1]), d=dt)

        # Filter only the positive frequencies
        positive_freqs = fft_freq > 0
        fft_result_positive = np.abs(fft_result[positive_freqs])
        
        return fft_result_positive

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

