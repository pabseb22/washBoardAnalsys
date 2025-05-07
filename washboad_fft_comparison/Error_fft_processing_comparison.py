import os
import numpy as np
from fft_comparison_cellbedform import CellBedform
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from frechetdist import frdist



# CONSTANTS

# TEST CASES 
TEST_CASES = [
    # 40 equivalente a 38.16
    # Base Reference
    {'velocity': '2.08ms', 'D': 1.2, 'Q': 0.2, 'L0': 5000, 'b': 40},

    # D variation
    {'velocity': '2.61ms', 'D': 0.4, 'Q': 0.2, 'L0': 5000, 'b': 40},
    # {'velocity': '2.61ms', 'D': 1.2, 'Q': 0.2, 'L0': 5000, 'b': 40},
    {'velocity': '2.61ms', 'D': 1.5, 'Q': 0.2, 'L0': 5000, 'b': 40},
]

# EXPERIMENTAL DATA FILES MANAGEMENT
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
BASE_SURFACE_FILE = "Vuelta5.txt"
SKIPROWS_FILES = 1
ALL_FFTS = []
TEST_FILE = "Variation_InitialSurface_Variation_Comparison_.xlsx"
FOLDER = os.path.join("Results",TEST_FILE)

# CELLBEDFORM NUMERICAL SIMULATION PARAMETERS
STEPS_CELLBEDFORM = 75
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

def run_test_cases(initial_surface,test_case):
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
    cb.run(STEPS_CELLBEDFORM)
    
    ALL_FFTS.append(cb.extract_experimental_fft())

def main():
    for _,test_case in enumerate(TEST_CASES, start=1):
        print("Starting test for: ",test_case['velocity'])
        # Create initial surface
        base_surface_file_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, test_case['velocity'], BASE_SURFACE_FILE)
        base_surface_exp_data = load_experimental_data(base_surface_file_path)
        initial_surface = create_initial_surface(base_surface_exp_data)

        # Run test cases
        run_test_cases(initial_surface,test_case)
        
    error_percentages = []
    frechet_distances = []

    # Obtener la referencia base de la primera FFT
    base_fft_freq, base_fft_result = ALL_FFTS[0][:2]  # Usamos la primera FFT como base (frecuencia y resultado de FFT)

    for i, fft_data in enumerate(ALL_FFTS):
        fft_freq, fft_result = fft_data[:2]  # Solo consideramos la frecuencia y el resultado de la FFT

        if i == 0:
            continue  # La primera FFT se usa como referencia

        print(f"Comparando caso de prueba {i+1} con la referencia base...")

        # -------- Comparación de las FFTs --------
        # Interpolación de la frecuencia de la base para que coincida con la frecuencia de la FFT numérica
        interp_base = interp1d(base_fft_freq, np.abs(base_fft_result), bounds_error=False, fill_value="extrapolate")
        fft_base_interp = interp_base(fft_freq)

        # -------- Error Porcentual de las Integrales --------
        # Usamos la integral en el dominio de la frecuencia
        integral_base = np.trapz(np.abs(fft_base_interp), fft_freq)
        integral_fft = np.trapz(np.abs(fft_result), fft_freq)
        integral_error = np.abs((integral_base - integral_fft) / integral_base) * 100
        error_percentages.append(integral_error)

        # # -------- Distancia de Fréchet --------
        # # Calculamos la distancia de Fréchet entre las FFTs en el dominio de la frecuencia
        # freq_points = np.column_stack((fft_freq, np.abs(fft_result)))
        # base_freq_points = np.column_stack((fft_freq, np.abs(fft_base_interp)))
        # frechet_dist = frdist(freq_points, base_freq_points)
        # frechet_distances.append(frechet_dist)

    # Verificación de longitud de las listas antes de graficar
    print(f"Length of error_percentages: {len(error_percentages)}")
    print(f"Length of frechet_distances: {len(frechet_distances)}")
    print(f"Length of ALL_FFTS: {len(ALL_FFTS)}")

    if len(error_percentages) != len(ALL_FFTS) - 1:
        print("Error: Las listas no tienen la misma longitud. Revisar cálculo de errores.")
        return

    # Graficar error porcentual
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ALL_FFTS)), error_percentages, marker='^', color='green')
    plt.title('Error Porcentual de las Integrales por Caso de Prueba')
    plt.xlabel('Caso de Prueba')
    plt.ylabel('Error (%)')
    plt.grid()

    # Graficar distancias de Fréchet
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(ALL_FFTS)), frechet_distances, marker='s', color='red')
    # plt.title('Distancia de Fréchet por Caso de Prueba')
    # plt.xlabel('Caso de Prueba')
    # plt.ylabel('Distancia de Fréchet')
    # plt.grid()

    plt.show()

    # Guardar las métricas de comparación en un archivo de Excel
    summary_df = pd.DataFrame({
        'Test Case': list(range(1, len(ALL_FFTS))),
        'Error % Integrales': error_percentages,
        'Frechet Distance': frechet_distances
    })

    summary_df.to_excel(FOLDER.replace('.xlsx', '_Summary.xlsx'), index=False)

if __name__ == "__main__":
    main()