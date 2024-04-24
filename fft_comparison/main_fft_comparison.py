import datetime
from fft_comparison_cellbedform import CellBedform
import os
import numpy as np
from scipy.interpolate import interp1d

print("Program Initialization")
program_start_time = datetime.datetime.now()

steps = 75
save_space = 20
save_values = list(range(1, steps + 1, save_space))
print("Steps to be saved:")
print(save_values)

file_path = os.path.join("ExperimentalData", "5thPass2ms.txt") #Esta en mm Los datos de altura y el x esta en metros
data_5th_pass = np.loadtxt(file_path)
dx = len(data_5th_pass)
dy = 40
y_cut = 20
# Define same original surface to get consistent results
interpolated_profile = interp1d(data_5th_pass[:, 0].T*1000, data_5th_pass[:, 1].T)
data_exp = interpolated_profile(np.arange(1, dx+1, 1))

initial_surface = np.tile(data_exp[:, np.newaxis], (1, dy))

test_cases = [
    #{'name': 'C_1', 'D': 1.2, 'Q': 0.2, 'L0': 1000, 'b': 38, }, #Mismo periodo
    {'name': 'C_1', 'D': 1.2, 'Q': 0.2, 'L0': 3489.10770983, 'b':  51.02204329, }, #Test
]

#Excelente: {'name': 'C_1', 'D': 1.2, 'Q': 0.2, 'L0': 1000, 'b': 38, }
#912.31250763  41.32538414]




for idx, test_case in enumerate(test_cases, start=1):
    iteration_start_time = datetime.datetime.now()
    print(f"Starting Test For: {test_case['name']} at {iteration_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    cb = CellBedform(grid=(dx, dy), D=test_case['D'], Q=test_case['Q'], L0=test_case['L0'], b=test_case['b'], y_cut=y_cut,h=initial_surface)
    cb.run(steps, save_values, folder=test_case['name'])
    # cb.save_images(folder=test_case['name'], filename=f"{test_case['name']}_case_{idx}", save_steps=save_values)
    cb.compare_fft(save_values, folder=test_case['name'])
    iteration_end_time = datetime.datetime.now()
    print(f"Finished Test For: {test_case['name']} at {iteration_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = (iteration_end_time - iteration_start_time).total_seconds() / 60
    print(f"Duration: {duration:.2f} minutes")
    print("")

program_end_time = datetime.datetime.now()
total_duration = (program_end_time - program_start_time).total_seconds() / 60
print(f"Program Ended in: {total_duration:.2f} minutes")

