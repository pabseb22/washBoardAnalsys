import os
import numpy as np
from fft_comparison_cellbedform import CellBedform
from scipy.interpolate import interp1d

# CONSTANTS

# TEST CASES
TEST_CASES = [
    {'velocity': '0.78ms', 'D': 1.2, 'Q': 0.2, 'L0': -40.39188802, 'b': 45.82976483},
    # {'velocity': '1.29ms', 'D': 1.2, 'Q': 0.2, 'L0': 4588.61355303, 'b': 55.65060651},
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4837.67381703, 'b': 48.26318216},
    # {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 4986.31878445, 'b': 50.22157854},
]

# EXPERIMENTAL DATA FILES MANAGEMENT
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
BASE_SURFACE_FILE = "Vuelta5.txt"
EXPERIMENTAL_COMPARISON_FILE = "Vuelta80_filtered.txt"
SKIPROWS_FILES = 1

# CELLBEDFORM NUMERICAL SIMULATION PARAMETERS
STEPS_CELLBEDFORM = 75
D_Y = 40
D_X = 4450
Y_CUT = 20

def load_experimental_data(file_path):
    """Load and preprocess experimental data obtaining its fft and interpolating it to 4450 mm."""
    data = np.loadtxt(file_path, skiprows=SKIPROWS_FILES) # Load file
    offset = np.mean(data[:, 1]) # Center the Signal on the axis
    data[:, 1] -= offset
    data[:,0]=data[:,0]*1000-min(data[:,0])*1000 # Normalize and transforming to mm
    f = interp1d(data[:,0], data[:,1]) # Interpolation
    array = np.arange(0, D_X, 1)
    y_inter=f(array)
    data_inter=np.array([array,y_inter])
    return data_inter

def create_initial_surface(data_surface):
    """Create the initial surface for simulation."""
    data_exp = data_surface[1]  # Use the second column as the data
    return np.tile(data_exp[:, np.newaxis], (1, D_Y))

def run_test_cases(initial_surface, experimental_comparison_data,test_case):
    """Run test cases and compare FFT results."""
    cb = CellBedform(
        grid=(D_X, D_Y),
        D=test_case['D'],
        Q=test_case['Q'],
        L0=test_case['L0'],
        b=test_case['b'],
        y_cut=Y_CUT,
        h=initial_surface
    )
    cb.run(STEPS_CELLBEDFORM)
    cb.compare_fft(experimental_comparison_data, test_case['velocity'])

def main():
    for _,test_case in enumerate(TEST_CASES, start=1):
        print("Starting test for: ",test_case['velocity'])
        # Create initial surface
        base_surface_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, test_case['velocity'], BASE_SURFACE_FILE)
        base_surface_exp_data = load_experimental_data(base_surface_file_path)
        initial_surface = create_initial_surface(base_surface_exp_data)

        # Load experimental data  
        experimental_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, test_case['velocity'], EXPERIMENTAL_COMPARISON_FILE)

        experimental_comparison_data = load_experimental_data(experimental_file_path)
        # Run test cases
        run_test_cases(initial_surface, experimental_comparison_data,test_case)


if __name__ == "__main__":
    main()
