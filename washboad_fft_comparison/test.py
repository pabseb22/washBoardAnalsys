import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

# Assume you have arrays X (m), Z (elevation), and steps (passages)
# X.shape = (num_steps, num_x_points)
# Z.shape = (num_steps, num_x_points)

# Example dummy data (Replace with your real data)
num_steps = 300  # Number of steps (passages)
num_x_points = 100  # Number of x positions
x = np.linspace(0, 5, num_x_points)  # X values from 0 to 5 meters
steps = np.arange(num_steps)  # Number of passage

# Simulated elevation data (replace with actual Z values)
Z = np.sin(2 * np.pi * x[None, :] * 0.5) * np.exp(-steps[:, None] / 100)  # Example Z profile evolution

# Compute dz/dx gradient
dzdx = np.gradient(Z, x, axis=1)

# Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(x, steps, dzdx, levels=50, cmap='jet')

# Add colorbar
cbar = plt.colorbar(contour)
cbar.set_label('dz/dx')

# Labels and title
plt.xlabel('x (m)')
plt.ylabel('No. of passage')
plt.title('Evolution of dz/dx over time')

# Optional: Add a horizontal line (example at step 50)
plt.axhline(y=50, color='red', linewidth=1)

# Show plot
plt.show()
