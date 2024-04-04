import gc
import datetime
from convergence_cellbedform import CellBedform
import os
import numpy as np

print("Program Initialization")
program_start_time = datetime.datetime.now()

steps = 1000
save_space = 200
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

test_cases = [
     {'name': 'C_1', 'D': 0.8, 'Q': 0.6, 'L0': 18, 'b': 2, },
    #  {'name': 'C_1', 'D': 1.5, 'Q': 0.7, 'L0': 20, 'b': 1, }, Ultima
    #  {'name': 'C_1', 'D': 1.5, 'Q': 0.7, 'L0': 90, 'b': 2, }, La Mejor
    # {'name': 'C_1', 'D': 0.8, 'Q': 0.7, 'L0': 70, 'b': 2, }, Buenaza
    # {'name': 'C_1', 'D': 0.8, 'Q': 0.6, 'L0': 15, 'b': 2, }, Buena Amplitud
    # {'name': 'C_1', 'D': 0.2, 'Q': 0.05, 'L0': 10, 'b': 0.1, }, Mismo periodo
]

# Cada parametro incrementarlo con todos en 0.01 no hace practicamente nada excepto por el D, amplifica todo
# Q y b trabajan en conjunto
# Mucho D lo cuelga a todo
# En mismas condiciones un mayor L0 aumenta la amplitud



for idx, test_case in enumerate(test_cases, start=1):
    iteration_start_time = datetime.datetime.now()
    print(f"Starting Test For: {test_case['name']} at {iteration_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    cb = CellBedform(grid=(dx, dy), D=test_case['D'], Q=test_case['Q'], L0=test_case['L0'], b=test_case['b'], y_cut=y_cut,h=initial_surface)
    cb.run(steps, save_values, folder=test_case['name'])
    # cb.save_images(folder=test_case['name'], filename=f"{test_case['name']}_case_{idx}", save_steps=save_values)
    cb.plot_convergence(save_values, folder=test_case['name'])
    gc.collect()
    iteration_end_time = datetime.datetime.now()
    print(f"Finished Test For: {test_case['name']} at {iteration_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = (iteration_end_time - iteration_start_time).total_seconds() / 60
    print(f"Duration: {duration:.2f} minutes")
    print("")

program_end_time = datetime.datetime.now()
total_duration = (program_end_time - program_start_time).total_seconds() / 60
print(f"Program Ended in: {total_duration:.2f} minutes")


"""
        D : float, optional
            Diffusion coefficient for rolling and sliding transport.
            Larger values prescribes larger rates of tranport. Default
            value is 0.8.

        Q : float, optional
            Entrainment rate of saltation transport. Larger values prescribes
            the larger rate of sediment pick-up by flows. Default value is 0.6.

        L0 : float, optional
            Minimum length of saltation transport length. Default is 7.3.

        b : float, optional
            A coefficient to determine saltation length. Larger value
            prescribes longer transport length Default is 2.0.
"""
