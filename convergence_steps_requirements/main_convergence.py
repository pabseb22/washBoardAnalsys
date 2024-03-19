import gc
import datetime
from convergence_cellbedform import CellBedform

print("Program Initialization")
program_start_time = datetime.datetime.now()

steps = 30000
save_space = 600
save_values = list(range(1, steps + 1, save_space))
print("Steps to be saved:")
print(save_values)

test_cases = [
    {'name': 'C_1', 'Q': 0.01, 'L0': 1, 'b': 0.2, 'D': 0.2, 'dx': 1500, 'dy': 400, 'y_cut': 200, 'steps': steps, 'save_steps': save_values},
    {'name': 'C_2', 'Q': 0.6, 'L0': 7.3, 'b': 2, 'D': 0.8, 'dx': 1500, 'dy': 400, 'y_cut': 200, 'steps': steps, 'save_steps': save_values},
    {'name': 'C_3', 'Q': 0.1, 'L0': 2, 'b': 2, 'D': 0.8, 'dx': 1500, 'dy': 400, 'y_cut': 200, 'steps': steps, 'save_steps': save_values},
]



for idx, test_case in enumerate(test_cases, start=1):
    iteration_start_time = datetime.datetime.now()
    print(f"Starting Test For: {test_case['name']} at {iteration_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    cb = CellBedform(grid=(test_case['dx'], test_case['dy']), D=test_case['D'], Q=test_case['Q'], L0=test_case['L0'], b=test_case['b'], y_cut=test_case['y_cut'])
    cb.run(test_case['steps'], test_case['save_steps'], folder=test_case['name'])
    cb.save_images(folder=test_case['name'], filename=f"{test_case['name']}_case_{idx}", save_steps=test_case['save_steps'])
    cb.plot_convergence(test_case['save_steps'], folder=test_case['name'])
    
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
