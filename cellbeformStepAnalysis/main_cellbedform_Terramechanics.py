from cellbedform_Terramechanics import CellBedform

test_cases = [
    {'name':'C_1','Q': 0.6, 'L0': 7.3, 'b': 2, 'D': 0.8, 'dx': 150, 'dy': 40, 'y_cut': 20, 'steps':11,'save_steps':[1, 4, 7, 10]},
    #{'name':'C_2','Q': 0.6, 'L0': 7.3, 'b': 2, 'D': 0.8, 'dx': 150, 'dy': 40, 'y_cut': 20, 'steps':100,'save_steps':[1, 10, 50, 99]},
]

for idx, test_case in enumerate(test_cases, start=1):
    Q = test_case['Q']
    L0 = test_case['L0']
    b = test_case['b']
    D = test_case['D']
    dx = test_case['dx']
    dy = test_case['dy']
    y_cut = test_case['y_cut']
    folder_name = test_case['name']
    steps = test_case['steps']
    save_steps = test_case['save_steps']

    cb = CellBedform(grid=(dx, dy), D=D, Q=Q, L0=L0, b=b, y_cut=y_cut)
    cb.run(steps, save_steps)
    cb.save_images(folder=folder_name, filename=folder_name+f'_case_{idx}', save_steps=save_steps)

