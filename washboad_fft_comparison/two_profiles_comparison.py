import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- PARÁMETROS ---
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
BASE_SURFACE_FILE = "Vuelta80.txt"
COMPARISON_FILE = "R1.txt"
SKIPROWS_FILES = 1
D_X = 1500  # Longitud fija de comparación (mm)

# >>> Selección manual de punto de inicio de la ventana (en mm) <<<
X0_BASE = 200  # punto inicial de la curva base (en mm)
X0_COMP = 0   # punto inicial de la curva comparada (en mm)

def load_experimental_data(file_path):
    """Carga datos experimentales y centra el eje vertical restando el offset."""
    data = np.loadtxt(file_path, skiprows=SKIPROWS_FILES)
    offset = np.mean(data[:, 1])
    data[:, 1] -= offset
    data[:, 0] = data[:, 0] * 1000  # pasar x a mm
    return data

def extract_window(data, x0, length=D_X):
    """Extrae una ventana de longitud fija desde x0 y cambia eje X para que inicie en 0"""
    # Verifica el rango máximo de X en los datos comparados
    x_max = data[:, 0].max()

    # Asegúrate de que la ventana no se pase del rango
    if x0 + length > x_max:
        limit = x_max - x0
        print(f"⚠️  D_X ajustado a {limit:.2f} mm para no salir del rango de datos.")

    data = data[data[:, 0] >= x0]
    data = data[data[:, 0] <= x0 + length]
    data[:, 0] -= x0  # reposicionar eje X desde 0
    return data

def interpolate_uniform(data, step=1, length=D_X):
    """Interpola los datos para tener un espaciado uniforme cada 1 mm"""
    x_interp = np.arange(0, length, step)
    f_interp = interp1d(data[:, 0], data[:, 1], bounds_error=False, fill_value=0)
    y_interp = f_interp(x_interp)
    return np.array([x_interp, y_interp])

def calculate_fft(data):
    x = data[0]
    y = data[1]
    dx = np.mean(np.diff(x))
    fft_result = np.fft.fft(y) * dx
    fft_freq = np.fft.fftfreq(len(y), d=dx)
    return fft_freq, np.abs(fft_result)

def save_to_excel(base_interp, comp_interp, fft_base, fft_comp):
    """Guarda los datos en un archivo Excel."""

    # Organizar los datos en DataFrames
    base_data = np.vstack((base_interp[0], base_interp[1])).T  # Transponer para obtener la forma correcta
    comp_data = np.vstack((comp_interp[0], comp_interp[1])).T  # Transponer para obtener la forma correcta
    fft_base_data = np.vstack((fft_base[0], fft_base[1])).T  # Transponer para obtener la forma correcta
    fft_comp_data = np.vstack((fft_comp[0], fft_comp[1])).T  # Transponer para obtener la forma correcta

    # Crear DataFrames
    df_base = pd.DataFrame(base_data, columns=["X_Base", "Y_Base"])
    df_comp = pd.DataFrame(comp_data, columns=["X_Comp", "Y_Comp"])
    df_fft_base = pd.DataFrame(fft_base_data, columns=["Freq_Base", "FFT_Base"])
    df_fft_comp = pd.DataFrame(fft_comp_data, columns=["Freq_Comp", "FFT_Comp"])


    # Escribir todo en un archivo Excel
    with pd.ExcelWriter("pista_vs_dron_(2.08vsR1)_results.xlsx") as writer:
        df_base.to_excel(writer, sheet_name="Base", index=False)
        df_comp.to_excel(writer, sheet_name="Comparative", index=False)
        df_fft_base.to_excel(writer, sheet_name="FFT_Base", index=False)
        df_fft_comp.to_excel(writer, sheet_name="FFT_Comparative", index=False)




def main():
    # --- Cargar curvas crudas ---
    base_path = os.path.join("ExperimentalData", CONDITIONS_FOLDER, "2.08ms", BASE_SURFACE_FILE)
    comp_path = os.path.join("ExperimentalData", "Drones", COMPARISON_FILE)

    base_raw = load_experimental_data(base_path)
    comp_raw = load_experimental_data(comp_path)

    # alinear que el x sea el 0,0
    base_raw[:, 0] -= base_raw[0, 0]
    comp_raw[:, 0] -= comp_raw[0, 0]

    # --- Extraer ventanas ---
    base_window = extract_window(base_raw, X0_BASE)
    comp_window = extract_window(comp_raw, X0_COMP)

    # --- Interpolación uniforme ---
    base_interp = interpolate_uniform(base_window)
    comp_interp = interpolate_uniform(comp_window)

    # --- Calcular FFTs ---
    freq_base, fft_base = calculate_fft(base_interp)
    freq_comp, fft_comp = calculate_fft(comp_interp)

    # --- Integrales y diferencias ---
    int_base = np.trapz(fft_base, freq_comp)
    int_comp = np.trapz(fft_comp, freq_comp)
    abs_diff = np.abs(int_base - int_comp)
    rel_error = (abs_diff / int_base) * 100

    # --- Error cuadrático acumulado tipo loss function ---
    squared_diffs = (fft_base - fft_comp)**2
    mse = np.mean(squared_diffs)

    print("\n--- RESULTADOS INTEGRALES ---")
    print(f"Integral Base        : {int_base:.4f}")
    print(f"Integral Comparativa : {int_comp:.4f}")
    print(f"Diferencia absoluta  : {abs_diff:.4f}")
    print(f"Error porcentual     : {rel_error:.2f}%")

    print("\n--- RESULTADOS DIFERENCIA VALORES Y ---")
    print(f"Error cuadrático medio (MSE)     : {mse:.6f}")

    # Guardar los resultados en un archivo Excel
    save_to_excel(base_interp, comp_interp, [freq_base, fft_base], [freq_comp, fft_comp])

    # --- Graficar comparativa de ventanas (perfil) ---
    plt.figure(figsize=(10, 4))
    plt.plot(base_raw[:, 0], base_raw[:, 1], label="Base (2.08 m/s)")
    plt.plot(comp_raw[:, 0], comp_raw[:, 1], label=COMPARISON_FILE, alpha=0.8)
    plt.title("Ventana Perfiles")
    plt.xlabel("Distancia (mm)")
    plt.ylabel("Altura (mm)")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(10, 4))
    plt.plot(base_interp[0], base_interp[1], label="Base (2.08 m/s)")
    plt.plot(comp_interp[0], comp_interp[1], label=COMPARISON_FILE, alpha=0.8)
    plt.title("Ventana comparada (alineadas desde 0 hasta D_X)")
    plt.xlabel("Distancia (mm)")
    plt.ylabel("Altura (mm)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # --- Graficar FFTs ---
    plt.figure(figsize=(10, 4))
    plt.plot(freq_comp, fft_base, label="FFT Base 2.08ms", alpha=0.7)
    plt.plot(freq_comp, fft_comp, label=f"FFT {COMPARISON_FILE}", alpha=0.7)
    plt.title("Comparación de FFTs")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.xlim(0, 0.05)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
