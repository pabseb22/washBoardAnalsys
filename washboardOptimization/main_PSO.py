from pyswarms.single.global_best import GlobalBestPSO
from cellbedform_PSO import CellBedform
import numpy as np
import os, sys, datetime


# Constants
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
TEST_FOLDER = "0.78ms"
BASE_SURFACE_FILE = "Vuelta5.txt"
EXPERIMENTAL_COMPARISON_FILE = "Vuelta80.txt"
STEPS_CELLBEDFORM = 75
OPTIMIZATION_STEPS = 100
N_PARTICLES = 10
PSO_BOUNDS = (np.array([4000, 45]), np.array([500, 20])) 
PSO_OPTIONS = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
D_Y = 40
Y_CUT = 20
SKIPROWS_FILES = 1

def initialize_program():
    """Initialize the program and print the start time."""
    print("\nProgram Initialization")
    start_time = datetime.datetime.now()
    print(f"Time Initialization at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return start_time

def create_initial_surface(file_path):
    """Create the initial surface for simulation."""
    data_surface = np.loadtxt(file_path, skiprows=SKIPROWS_FILES)
    data_exp = data_surface[:, 1]  # Use the second column as the data
    global dx
    dx = len(data_exp)
    return np.tile(data_exp[:, np.newaxis], (1, D_Y))


def load_experimental_data(file_path):
    """Load and preprocess experimental data obtaining its fft."""
    data = np.loadtxt(file_path, skiprows=SKIPROWS_FILES) # Load file
    offset = np.mean(data[:, 1]) # Centering the Signal on the axis
    data[:, 1] -= offset
    position_values = data[:, 0]
    dt = np.mean(np.diff(position_values))  # Compute the average time step
    return data, dt

def perform_fft(data, dt):
    """Perform FFT on the experimental data."""
    fft_result = np.fft.fft(data[:, 1]) * dt
    fft_freq = np.fft.fftfreq(len(data[:, 1]), d=dt) * dt
    return np.abs(fft_result)

def objective_function(params):
    """Objective function to minimize."""
    global control, total_comparisons
    L0_ = params[:, 0]
    b_ = params[:, 1]
    differences = []
    print("")
    for L0, b in zip(L0_, b_):
        control += 1
        cb = CellBedform(grid=(dx, D_Y), D=1.2, Q=0.2, L0=L0, b=b, y_cut=Y_CUT, h=initial_surface)
        fft_numerical = cb.run(STEPS_CELLBEDFORM) # Perform Cellbedform Numerical Simulation and obtain fft
    
        min_length = min(len(fft_exp), len(fft_numerical))  # Ensure the comparison is done with the overlapping part
        fft_exp_slice = fft_exp[:min_length]
        fft_numerical_slice = fft_numerical[:min_length]
        
        peak_index = np.argmax(fft_exp_slice)
        margin = int(0.1 * len(fft_exp_slice))  # Identify 10% of the total amount of data next to the highest peak to ponderate
        start_index = max(0, peak_index - margin)
        end_index = min(len(fft_exp_slice), peak_index + margin)
        
        diff = fft_exp_slice - fft_numerical_slice
        weighted_diff = np.copy(diff)
        weighted_diff[start_index:end_index] *= 2  # Amplify error importance to 10% of data from the peak of FFT
        difference = np.sum(weighted_diff ** 2)
        differences.append(difference)
        print(f"{control}/{total_comparisons} -> [ {L0}, {b} ]")
    return np.array(differences)

def main():
    program_start_time = initialize_program()

    # Load experimental data
    experimental_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, TEST_FOLDER, EXPERIMENTAL_COMPARISON_FILE)
    base_surface_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, TEST_FOLDER, BASE_SURFACE_FILE)
    data_exp, dt = load_experimental_data(experimental_file_path)
    global fft_exp
    fft_exp = perform_fft(data_exp, dt)

    # Create initial surface
    global initial_surface
    initial_surface = create_initial_surface(base_surface_file_path)

    # Initialize PSO
    global control, total_comparisons
    control = 0
    total_comparisons = OPTIMIZATION_STEPS * N_PARTICLES
    optimizer = GlobalBestPSO(n_particles=N_PARTICLES, dimensions=2, options=PSO_OPTIONS, bounds=PSO_BOUNDS)

    # Optimize
    cost, pos = optimizer.optimize(objective_function, OPTIMIZATION_STEPS)

    # Display the result
    print("Best Position:", pos)

    program_end_time = datetime.datetime.now()
    total_duration = (program_end_time - program_start_time).total_seconds() / 60
    print(f"\nProgram Ended in: {total_duration:.2f} minutes")

if __name__ == "__main__":
    main()