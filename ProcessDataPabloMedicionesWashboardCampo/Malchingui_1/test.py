import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Crear una señal de ejemplo con frecuencias bajas y altas
fs = 500  # frecuencia de muestreo en Hz
t = np.linspace(0, 1, fs, endpoint=False)  # vector de tiempo de 1 segundo

# Crear una señal que mezcla una frecuencia baja (5 Hz) y una alta (50 Hz)
frecuencia_baja = 5
frecuencia_alta = 50
signal = np.sin(2 * np.pi * frecuencia_baja * t) + 0.5 * np.sin(2 * np.pi * frecuencia_alta * t)

# Diseñar el filtro pasa-alto
fc = 20  # frecuencia de corte (en Hz)
b, a = butter(4, fc / (0.5 * fs), btype='high')  # Filtro Butterworth de orden 4

# Aplicar el filtro a la señal
filtered_signal = filtfilt(b, a, signal)

# Graficar la señal original y la señal filtrada
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Señal Original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label='Señal Filtrada (High-Pass)', color='orange')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()

plt.tight_layout()
plt.show()
