from pyswarms.single.global_best import GlobalBestPSO
from cellbedform_PSO import CellBedform
from scipy.interpolate import interp1d
import numpy as np
import os, datetime

### CONSTANTS  ###
# EXPERIMENTAL DATA
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
TEST_FOLDERS = ["0.78ms","1.03ms"]
BASE_SURFACE_FILE = "Vuelta5.txt"
EXPERIMENTAL_COMPARISON_FILE = "Vuelta80_filtered.txt"
SKIPROWS_FILES = 1

# CELLBEDFORM NUMERICAL SIMULATION PARAMETERS
STEPS_CELLBEDFORM = 75
D_Y = 40
D_X = 4450 # Length Experimental Track in mm
Y_CUT = 20
# Parameters Used
D = 1.2 
Q = 0.2

# PSO OPTIMIZATION PARAMETERS
OPTIMIZATION_STEPS = 100
N_PARTICLES = 10
TOTAL_COMPARISONS = OPTIMIZATION_STEPS * N_PARTICLES
PSO_BOUNDS = (np.array([0, 20]),np.array([6000, 70])) 
PSO_OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# Estrategias disponibles
strategies = ['nearest', 'random', 'shrink', 'reflect', 'unmodified']

def initialize_program():
    """Initialize the program and print the start time."""
    print("\nProgram Initialization")
    start_time = datetime.datetime.now()
    print(f"Time Initialization at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return start_time

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

def perform_fft(data):
    """Perform FFT on the experimental data."""
    time_values = data[0]/1000
    dt = np.mean(np.diff(time_values))
    # Perform FFT on profile data
    fft_result = np.fft.fft(data[1]) * dt
    fft_freq = np.fft.fftfreq(len(data[1]), d=dt)

    # Filter only the positive frequencies
    positive_freqs = fft_freq > 0
    fft_result_positive = fft_result[positive_freqs]

    global fft_exp
    fft_exp = np.abs(fft_result)

def objective_function(params):
    """Objective function to minimize."""
    global control
    L0_ = params[:, 0]
    b_ = params[:, 1]
    differences = []
    print("")
    for L0, b in zip(L0_, b_):
        control += 1
        cb = CellBedform(grid=(D_X, D_Y), D=D, Q=Q, L0=L0, b=b, y_cut=Y_CUT, h=initial_surface)
        fft_numerical = cb.run(STEPS_CELLBEDFORM) # Perform Cellbedform Numerical Simulation and obtain fft
        difference = weighted_diff(fft_exp, fft_numerical)
        differences.append(difference)
        print(f"{control}/{TOTAL_COMPARISONS} -> [ {L0}, {b} ]")
    return np.array(differences)

def direct_diff(fft_exp,fft_numerical):
    diff = fft_exp - fft_numerical
    difference = np.sum(diff ** 2)
    return difference


def weighted_diff(fft_exp,fft_numerical):
    peak_index = np.argmax(fft_exp)
    margin = int(0.1 * len(fft_exp))  # Identify 10% of the total amount of data next to the highest peak to ponderate
    start_index = max(0, peak_index - margin)
    end_index = min(len(fft_exp), peak_index + margin)
    diff = fft_exp - fft_numerical
    diff[start_index:end_index] *= 2  # Amplify error importance to 10% of data from the peak of FFT
    difference = np.sum(diff ** 2)
    return difference

def main():
    program_start_time = initialize_program()

    results = []

    for TEST_FOLDER in TEST_FOLDERS:
        print(f"Running optimization for {TEST_FOLDER}")
        
        # Load experimental data
        experimental_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, TEST_FOLDER, EXPERIMENTAL_COMPARISON_FILE)
        base_surface_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, TEST_FOLDER, BASE_SURFACE_FILE)
        data_exp = load_experimental_data(experimental_file_path)

        perform_fft(data_exp)

        # Create initial surface
        base_surface_exp_data = load_experimental_data(base_surface_file_path)
        global initial_surface
        initial_surface = create_initial_surface(base_surface_exp_data)

        # Initialize PSO
        global control
        control = 0
        optimizer = GlobalBestPSO(n_particles=N_PARTICLES, dimensions=2, options=PSO_OPTIONS, bounds=PSO_BOUNDS, bh_strategy='shrink')

        # Optimize
        _, pos = optimizer.optimize(objective_function, OPTIMIZATION_STEPS)

        # Store the result
        results.append((TEST_FOLDER, pos))

        # Display the result
        print(f"Best Position for {TEST_FOLDER}: {pos}")

    program_end_time = datetime.datetime.now()
    total_duration = (program_end_time - program_start_time).total_seconds() / 60
    print(f"\nProgram Ended in: {total_duration:.2f} minutes")

if __name__ == "__main__":
    main()