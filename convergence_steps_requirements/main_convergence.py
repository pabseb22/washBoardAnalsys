import gc
import datetime
from convergence_cellbedform import CellBedform
import os
import numpy as np

print("Program Initialization")
program_start_time = datetime.datetime.now()

steps = 300
save_space = 60
save_values = list(range(1, steps + 1, save_space))
print("Steps to be saved:")
print(save_values)

file_path = os.path.join("ExperimentalData", "5thPass2ms.txt")
data_exp = np.loadtxt(file_path)
data_exp = data_exp[:, 1].T
dx = len(data_exp)
dy = 40
y_cut = 20
# Define same original surface to get consistent results
initial_surface = np.tile(data_exp[:, np.newaxis], (1, dy))

print(initial_surface)
test_cases = [
    {'name': 'C_1', 'Q': 0.1, 'L0': 1, 'b': 0.2, 'D': 0.2},
    {'name': 'C_2', 'Q': 0.1, 'L0': 2, 'b': 0.2, 'D': 0.2},
]



for idx, test_case in enumerate(test_cases, start=1):
    iteration_start_time = datetime.datetime.now()
    print(f"Starting Test For: {test_case['name']} at {iteration_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    cb = CellBedform(grid=(dx, dy), D=test_case['D'], Q=test_case['Q'], L0=test_case['L0'], b=test_case['b'], y_cut=y_cut,h=initial_surface)
    cb.run(steps, save_values, folder=test_case['name'])
    cb.save_images(folder=test_case['name'], filename=f"{test_case['name']}_case_{idx}", save_steps=save_values)
    cb.plot_convergence(save_values, folder=test_case['name'])
    gc.collect()
    iteration_end_time = datetime.datetime.now()
    print(f"Finished Test For: {test_case['name']} at {iteration_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = (iteration_end_time - iteration_start_time).total_seconds() / 60
    print(f"Duration: {duration:.2f} minutes")
    print("")

program_end_time = datetime.datetime.now()
total_duration = (program_end_time - program_start_time).total_seconds() / 60
print("")
print("")
print(f"Program Ended in: {total_duration:.2f} minutes")
