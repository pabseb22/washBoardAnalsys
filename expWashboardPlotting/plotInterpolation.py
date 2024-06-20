import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import os

# EXPERIMENTAL DATA FILES MANAGEMENT
CONDITIONS_FOLDER = "1200g_VelocidadVariable_1740kg-m3"
TEST_FOLDER = "0.78ms"
FILE = "Vuelta80.txt"

# Load the data into a NumPy array
filename = os.path.join("ExperimentalData", CONDITIONS_FOLDER,TEST_FOLDER,FILE)
print(filename)
Data = np.loadtxt(filename, skiprows=1)

Data[:,0]=Data[:,0]*1000-min(Data[:,0])*1000

f = interp1d(Data[:,0], Data[:,1])

array = np.arange(0, 4450, 1)
y_inter=f(array)

Data_inter=np.array([array,y_inter])

# Separate the x and y values

plt.plot(Data[:,0], Data[:,1],Data_inter[:,0],Data_inter[:,1])
plt.xlabel('X(mm)')
plt.ylabel('Z(mm)')
plt.title('Plot of X vs Y')
plt.grid(True)
plt.show()


