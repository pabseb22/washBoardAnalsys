import os
import numpy as np
from fft_comparison_cellbedform import CellBedform
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# CONSTANTS

# TEST CASES
TEST_CASES = [
    {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 4978.56, 'b': 38.16},
    {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 5000, 'b': 38.16},
    {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 4000, 'b': 38.16},
    {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 10000, 'b': 38.16},
    {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 1000, 'b': 38.16},

]

# EXPERIMENTAL DATA FILES MANAGEMENT
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
BASE_SURFACE_FILE = "Vuelta5.txt"
EXPERIMENTAL_COMPARISON_FILE = "Vuelta80.txt"
SKIPROWS_FILES = 1
ALL_FFTS = []
TEST_FILE = "L0_Variation_Comparison_.xlsx"
FOLDER = os.path.join("Results",TEST_FILE)

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

def run_test_cases(initial_surface,test_case):
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
    ALL_FFTS.append(cb.extract_experimental_fft())

def main():
    for _,test_case in enumerate(TEST_CASES, start=1):
        print("Starting test for: ",test_case['velocity'])
        # Create initial surface
        base_surface_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, test_case['velocity'], BASE_SURFACE_FILE)
        base_surface_exp_data = load_experimental_data(base_surface_file_path)
        initial_surface = create_initial_surface(base_surface_exp_data)

        # Run test cases
        run_test_cases(initial_surface,test_case)
    
    plt.close()
    # Plotting all FFT results on the same plot
    plt.figure(figsize=(10, 6))

    # Save Data
    # Prepare data for Excel
    data = {}
    for i, fft_data in enumerate(ALL_FFTS):
        fft_freq, fft_result, x_profile, y_profile = fft_data
        
        # Create column names with test case numbers
        data[f'Test Case {i+1} FFT Frequency'] = fft_freq
        data[f'Test Case {i+1} FFT Result'] = fft_result
        data[f'Test Case {i+1} X Profile'] = x_profile
        data[f'Test Case {i+1} Y Profile'] = y_profile

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(FOLDER, index=False)

    colors = ['green', 'blue', 'orange', 'purple', 'cyan']  # Add more colors if needed
    for i, fft_data in enumerate(ALL_FFTS):
        fft_freq, fft_result, x_profile, y_profile = fft_data
        color = colors[i % len(colors)]
        plt.plot(fft_freq, fft_result, color=color, label=f'FFT {i+1}')
    plt.xlim(0, 0.015)
    plt.title('Experimental FFTs')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    
    for i, fft_data in enumerate(ALL_FFTS):
        fft_freq, fft_result, x_profile, y_profile = fft_data
        color = colors[i % len(colors)]
        plt.plot(x_profile, y_profile, color=color, label=f'FFT {i+1}')
    plt.ylim(-25, 25)
    plt.title('Numerical Profiles')
    plt.xlabel('mm')
    plt.ylabel('mm')
    plt.legend()
    plt.grid(True)
    plt.show()





if __name__ == "__main__":
    main()
