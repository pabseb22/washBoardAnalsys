import os
import numpy as np
from fft_comparison_cellbedform import CellBedform
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# CONSTANTS

# TEST CASES
TEST_CASES = [
    # Velocidad Variable a 1740
    # {'velocity': '0.78ms', 'D': 1.2, 'Q': 0.2, 'L0': -70.89, 'b': 68.99, 'boundaries': [6, 25], 'min_distance': 200, 'low_pass':0.02,'control_steps':[5,15,35,75], 'save_images':False, 'compare_fft': True,'obtain_amplitude':True ,'obtain_scalogram': False},
    # {'velocity': '1.03ms', 'D': 1.2, 'Q': 0.2, 'L0': -50.67, 'b': 51.23, 'boundaries': [7, 37], 'min_distance': 200, 'low_pass':0.02,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':True ,'obtain_scalogram': False },
    # {'velocity': '1.29ms', 'D': 1.2, 'Q': 0.2, 'L0': 169.1, 'b': 49.73, 'boundaries': [14, 36], 'min_distance': 200, 'low_pass':0.008,'control_steps':[5, 100, 1000, 5000, 9999], 'save_images':False,'compare_fft': False,'obtain_amplitude':False ,'obtain_scalogram': False},
    # {'velocity': '1.55ms', 'D': 1.2, 'Q': 0.2, 'L0': 1026.41, 'b': 32.85, 'boundaries': [11, 33], 'min_distance': 200, 'low_pass':0.008,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': True,'obtain_amplitude':True ,'obtain_scalogram': False},
    {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4808.85, 'b': 50.94, 'boundaries': [4, 32], 'min_distance': 250, 'low_pass':0.008,'control_steps':list(range(1,100000,100)), 'save_images':False,'compare_fft': False,'obtain_amplitude':False ,'obtain_scalogram': False},
    # {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 4978.56, 'b': 38.16, 'boundaries': [4, 29], 'min_distance': 250, 'low_pass':0.008,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': True,'obtain_amplitude':True ,'obtain_scalogram': False},
    # {'velocity': '3.15ms', 'D': 1.2, 'Q': 0.2, 'L0': 1466.33, 'b': 67.90, 'boundaries': [3, 27], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': True,'obtain_amplitude':True ,'obtain_scalogram': False},
    
    # Velocidad Variable a 1520
    # {'velocity': '0.52ms', 'D': 1.2, 'Q': 0.2, 'L0': -62.33, 'b': 75.41, 'boundaries': [8, 22], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },
    # {'velocity': '1.03ms', 'D': 1.2, 'Q': 0.2, 'L0': -60.62, 'b': 40.24, 'boundaries': [8, 22], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },
    # {'velocity': '1.55ms', 'D': 1.2, 'Q': 0.2, 'L0': 2590.62, 'b': 30.03, 'boundaries': [8, 22], 'min_distance': 250, 'low_pass':0.011,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': True,'obtain_amplitude':True,'obtain_scalogram': False },
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4983.85, 'b': 30.94, 'boundaries': [8, 22], 'min_distance': 250, 'low_pass':0.011,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': True,'obtain_amplitude':True,'obtain_scalogram': False },
    # {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 5181.56, 'b': 18.86, 'boundaries': [8, 22], 'min_distance': 260, 'low_pass':0.025,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':True,'obtain_scalogram': False },

    # Masa Variable a 1740
    # {'velocity': '1200g', 'D': 1.2, 'Q': 0.2, 'L0': 4808.85, 'b': 50.94, 'boundaries': [8, 22], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': True,'obtain_amplitude':True,'obtain_scalogram': False },
    # {'velocity': '1331g', 'D': 1.2, 'Q': 0.2, 'L0': 4996.16, 'b': 46.69, 'boundaries': [8, 22], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': True,'obtain_amplitude':True,'obtain_scalogram': False },
    # {'velocity': '1475g', 'D': 1.2, 'Q': 0.2, 'L0': 4794.845, 'b': 47.38, 'boundaries': [8, 22], 'min_distance': 220, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },

    # Masa Variable a 1520
    # {'velocity': '1200g', 'D': 1.2, 'Q': 0.2, 'L0': 5108.85, 'b': 30.94, 'boundaries': [8, 22], 'min_distance': 300, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': False,'obtain_amplitude':True,'obtain_scalogram': False },
    # {'velocity': '1331g', 'D': 1.2, 'Q': 0.2, 'L0': 6794.85, 'b': 28.38, 'boundaries': [8, 22], 'min_distance': 300, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':True,'compare_fft': False,'obtain_amplitude':True,'obtain_scalogram': False },
    # {'velocity': '1475g', 'D': 1.2, 'Q': 0.2, 'L0': 4894.76, 'b': 46.27, 'boundaries': [8, 22], 'min_distance': 300, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },

    ###### OPCIONES ADICIONALES ######

    # Opciones 2.08ms
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 3866.94, 'b': 36.71, 'boundaries': [4, 32], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False ,'obtain_scalogram': False},
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4837.67, 'b': 48.26, 'boundaries': [4, 32], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False ,'obtain_scalogram': False},
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4808.85, 'b': 50.94, 'boundaries': [4, 32], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False ,'obtain_scalogram': False},
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4790.50, 'b': 41.58, 'boundaries': [4, 32], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False ,'obtain_scalogram': False},
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4074.97, 'b': 25.04, 'boundaries': [4, 32], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },

    # Previo 2.08ms
    # {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 4374.97, 'b': 19.04, 'boundaries': [4, 32], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False ,'obtain_scalogram': False},

    # Adicionales Masa Variable 1520
    # {'velocity': '1331g', 'D': 1.2, 'Q': 0.2, 'L0': 6824.85, 'b': 27.38, 'boundaries': [8, 22], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },

    # Adicionales Densidad Baja Velocidad Variable 1520
    # {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 5731.01, 'b': 37.40, 'boundaries': [8, 20], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },
    # {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 5178.56, 'b': 18.16, 'boundaries': [8, 20], 'min_distance': 200, 'low_pass':0.2,'control_steps':[5,15,35,75], 'save_images':False,'compare_fft': True,'obtain_amplitude':False,'obtain_scalogram': False },
]

# Objetivos: 
# Densidad Variable: 1520 debe ser una proyeccion superior de la actual 1740 por picos mas altos
# Masa Variable: 1740 debe usar 2.08 que tenemos para 1200 y tener uno mayor para 1331 e intermedio a 1475
# Masa Variable 2: 1520 debe tener todo por sobre 1740 de L0.


# EXPERIMENTAL DATA FILES MANAGEMENT
DENSITY = "1740kg-m3"
# CONDITIONS = "2.08ms_MasaVariable_"
CONDITIONS = "1200g_VelocidadVariable_"
CONDITIONS_FOLDER = CONDITIONS + DENSITY
BASE_SURFACE_FILE = "Vuelta5.txt"
EXPERIMENTAL_COMPARISON_FILE = "Vuelta80.txt"
RESULTS_FOLDER = "Results"
SKIPROWS_FILES = 1

# CELLBEDFORM NUMERICAL SIMULATION PARAMETERS
STEPS_CELLBEDFORM = 100001
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

def run_test_cases(initial_surface, experimental_comparison_data,test_case):
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

    folder = ""
    folder_name = str(test_case['velocity']+"_L0="+str(test_case['L0'])+"_b="+str(test_case['b'])+"_"+DENSITY)
    folder = os.path.join(RESULTS_FOLDER, folder_name)
    if(test_case['save_images']):
        # Create the main folder if it doesn't exist
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        # Create the main folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
    
    # cb.run(STEPS_CELLBEDFORM)
    cb.run_average_amplitude(STEPS_CELLBEDFORM, test_case['min_distance'], test_case['low_pass'], test_case['control_steps'], folder )

    if test_case['compare_fft']:
        cb.compare_fft(experimental_comparison_data, folder,test_case['boundaries'], test_case['control_steps'], test_case['save_images'])
    if test_case['obtain_amplitude']:
        cb.obtain_average_amplitude(test_case['min_distance'], test_case['low_pass'], test_case['control_steps'], folder, test_case['save_images'])
    if test_case['obtain_scalogram']:
        cb.plot_scalogram( folder,test_case['velocity'], False)

def main():
    for _,test_case in enumerate(TEST_CASES, start=1):
        print("Starting test for: ",test_case['velocity'])
        # Create initial surface
        base_surface_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, test_case['velocity'], BASE_SURFACE_FILE)
        base_surface_exp_data = load_experimental_data(base_surface_file_path)
        initial_surface = create_initial_surface(base_surface_exp_data)

        # Load experimental data  
        experimental_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, test_case['velocity'], EXPERIMENTAL_COMPARISON_FILE)

        experimental_comparison_data = load_experimental_data(experimental_file_path)
        # Run test cases
        run_test_cases(initial_surface, experimental_comparison_data,test_case)
    
    plt.show()


if __name__ == "__main__":
    main()
