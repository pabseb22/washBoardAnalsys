import matplotlib.pyplot as plt
import os

# Simplified dictionary to store optimization results
optimization_results = {
    0.78: {
        "best_cost": 17.604112413590595,
        "L0": -54.08357045 ,
        "b": 57.64161001
    },
    1.29: {
        "best_cost": 472.5288941019848,
        "L0": 4588.61355303,
        "b": 55.65060651
    },
    2.08: {
        "best_cost": 372.639275065379,
        "L0": 4837.67381703,
        "b": 48.26318216
    },
    2.61: {
        "best_cost": 339.71329700953237,
        "L0": 4986.31878445,
        "b": 50.22157854
    }
}


# Extract velocities, L0, and b from the dictionary
velocities = list(optimization_results.keys())
L0_values = [v["L0"] for v in optimization_results.values()]
b_values = [v["b"] for v in optimization_results.values()]

# Set font properties for the plots
plt.rcParams['font.family'] = 'Times New Roman'

# Plot L0 vs Velocity
plt.figure(figsize=(16/2.54, 11/2.54), dpi=200)  # Improved quality with higher dpi
plt.subplot(1, 2, 1)
plt.plot(velocities, L0_values, marker='o', linestyle='-', color='b')
plt.xlabel('Velocity (m/s)')
plt.ylabel('L0')
plt.title('L0', fontweight='bold')
plt.grid(True)

# Plot b vs Velocity
plt.subplot(1, 2, 2)
plt.plot(velocities, b_values, marker='o', linestyle='-', color='r')
plt.xlabel('Velocity (m/s)')
plt.ylim(0,80)
plt.ylabel('b')
plt.title('b', fontweight='bold')
plt.grid(True)

# Save the plot
output_file = os.path.join("Images", 'optimization_results.png')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
