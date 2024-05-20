import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from cellbedform_PSO import CellBedform
from scipy.interpolate import interp1d
import os
import datetime
import matplotlib.pyplot as plt

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution

print("\nProgram Initialization")
program_start_time = datetime.datetime.now()

print(f"Time Initialization at {program_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Calculate FFT for experimental data
file_path = os.path.join("ExperimentalData", "80thPass2ms.txt")
data_exp = np.loadtxt(file_path)
data_exp_offset = np.mean(data_exp[:, 1])
data_exp[:, 1] = data_exp[:, 1] - data_exp_offset
position_values = data_exp[:, 0]
dt = np.mean(np.diff(position_values))  # Compute the average time step

# Perform FFT on experimental data
fft_result = np.fft.fft(data_exp[:, 1])*dt
fft_freq = np.fft.fftfreq(len(data_exp[:, 1]), d=dt)*dt
fft_exp = np.abs(fft_result)



# Data of 5th Pass to generate same initial surface
file_path_2 = os.path.join("ExperimentalData", "5thPass2ms.txt")
data_surface = np.loadtxt(file_path_2)
dx = len(data_surface)
dy = 40
y_cut = 20

# Define same original surface to get consistent results
interpolated_profile = interp1d(data_surface[:, 0].T*1000, data_surface[:, 1].T)
data_exp = interpolated_profile(np.arange(1, dx+1, 1))
initial_surface = np.tile(data_exp[:, np.newaxis], (1, dy))

# Steps to be analized in Cellbedform Simulation
steps = 75

# Steps for PSO optimization 
optimization_steps = 5
# Number of particles to be analized in each step of PSO optimization
n_particles = 10

# Flags to control current progress in Optmization
control = 0
total_comparisons = optimization_steps*n_particles #100 -> 11 minutes on i5

# Objective function to minimize 
def objective_function(params):
    print("")
    global control, total_comparisons
    iteration_start_time = datetime.datetime.now()
    L0_ = params[:, 0]
    b_ = params[:, 1]
    differences = []
    for L0, b in zip(L0_, b_):
        control += 1
        cb = CellBedform(grid=(dx, dy), D=1.2, Q=0.2, L0=L0, b=b, y_cut=y_cut, h=initial_surface)
        fft_num = cb.run(steps)

        # Calculate the center region (10% margin from the peak)
        peak_index = np.argmax(fft_exp)
        margin = int(0.1 * len(fft_exp))
        start_index = max(0, peak_index - margin)
        end_index = min(len(fft_exp), peak_index + margin)

        # Calculate the differences
        diff = fft_exp - fft_num

        # Apply additional weight to the center part of the FFT
        weighted_diff = np.copy(diff)
        weighted_diff[start_index:end_index] *= 2

        difference = np.sum(weighted_diff**2)  # Sum of squared differences
        differences.append(difference)
        print(f"{control}/{total_comparisons} -> [ {L0}, {b} ]")
    print(f"\nTime {iteration_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return np.array(differences)



#Define the bounds for each parameter
x_max=np.array([2000, 45])
x_min=np.array([500, 30])
bounds = (x_max, x_min)  # Example bounds, adjust as needed

# Call the optimizer
#Initialize the swarm
# Define the options for PSO optimizer https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html#module-pyswarms.single.global_best
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Create a Particle Swarm Optimizer
optimizer = GlobalBestPSO( n_particles=10, dimensions=2, options=options, bounds=bounds)

cost, pos = optimizer.optimize(objective_function, optimization_steps)

# Display the result
print("Best Position:", pos)

program_end_time = datetime.datetime.now()
total_duration = (program_end_time - program_start_time).total_seconds() / 60
print("")
print("")
print(f"Program Ended in: {total_duration:.2f} minutes")
