import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from cellbedform_PSO import CellBedform
from scipy.interpolate import interp1d
import os
import datetime
import matplotlib.pyplot as plt

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution

print("Program Initialization")
program_start_time = datetime.datetime.now()

print(f"Time Initialization at {program_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate FFT for experimental data
file_path = os.path.join("ExperimentalData", "80thPass2ms.txt")
data_exp = np.loadtxt(file_path)
data_exp_offset = np.mean(data_exp[:, 1])
data_exp[:, 1] = data_exp[:, 1] - data_exp_offset
time_values = data_exp[:, 0]
dt = np.mean(np.diff(time_values))  # Compute the average time step

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

# Steps to be analized
steps = 80

# Objective function to minimize 
def objective_function(params):
    iteration_start_time = datetime.datetime.now()
    print("")
    print(f"Time Step at {iteration_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    print("Tested Params: ")
    print(params)
    L0_ = params[:, 0]
    b_ = params[:, 1]
    differences = []
    control = 0
    for L0, b in zip(L0_, b_):
        control += 1
        print("Finished Particle #",control)
        cb = CellBedform(grid=(dx, dy), D=1.2, Q=0.2, L0=L0, b=b, y_cut=y_cut, h=initial_surface)
        fft_num = cb.run(steps)
        diff = fft_exp - fft_num
        difference = np.sum(diff**2)
        differences.append(difference)
    return np.array(differences)



#Define the bounds for each parameter
bounds = [(500, 2000), (30, 45)]  # Example bounds, adjust as needed

# Call the optimizer
#Initialize the swarm
# Define the options for PSO optimizer
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Create a Particle Swarm Optimizer
optimizer = GlobalBestPSO( n_particles=10, dimensions=2, options=options, bounds=bounds)

cost, pos = optimizer.optimize(objective_function, 1)

# Display the result
print("Best Position:", cost)

program_end_time = datetime.datetime.now()
total_duration = (program_end_time - program_start_time).total_seconds() / 60
print("")
print("")
print(f"Program Ended in: {total_duration:.2f} minutes")
