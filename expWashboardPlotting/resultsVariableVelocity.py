import matplotlib.pyplot as plt
import os

# Simplified dictionary to store optimization results
optimization_results = {
    0.78: {
        "velocity": "0.78ms",
        "D": 1.2,
        "Q": 0.2,
        "L0": -40.39188802,
        "b": 45.82976483
    },
    1.03: {
        "velocity": "1.03ms",
        "D": 1.2,
        "Q": 0.2,
        "L0": -2299.30289414,
        "b": 45.85475005
    },
    1.29: {
        "velocity": "1.29ms",
        "D": 1.2,
        "Q": 0.2,
        "L0": 4647.05717735,
        "b": 68.69560751
    },
    1.55: {
        "velocity": "1.55ms",
        "D": 1.2,
        "Q": 0.2,
        "L0": 5844.33512412,
        "b": 51.92830198
    },
    2.08: {
        "velocity": "2.08ms",
        "D": 1.2,
        "Q": 0.2,
        "L0": 5911.81381374,
        "b": 60.81679517
    },
    2.61: {
        "velocity": "2.61ms",
        "D": 1.2,
        "Q": 0.2,
        "L0": 5699.00532478,
        "b": 58.49857392
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
plt.ylabel('b')
plt.title('b', fontweight='bold')
plt.grid(True)

# Save the plot
output_file = os.path.join("Images", 'optimization_results.png')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()
