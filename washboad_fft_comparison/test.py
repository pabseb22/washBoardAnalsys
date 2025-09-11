from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# Puntos base aleatorios (menos que los reales)
x_base = np.linspace(0.0005, 0.0073, 6)
y_base = np.random.uniform(-2, 2, size=len(x_base))

# Interpolación para obtener zg
interp_func = interp1d(x_base, y_base, kind='cubic')
x = np.linspace(0.0005, 0.0073, 18)
zg = interp_func(x)

plt.plot(x, zg, marker='o')
plt.title("Superficie interpolada aleatoria")
plt.xlabel("x")
plt.ylabel("zg")
plt.grid(True)
plt.show()
