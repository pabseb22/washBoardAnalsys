import numpy as np
import pyswarms as ps

# Objective function to minimize (quadratic function)
def objective_function(x):
    return np.sum(x**2)

# Initialize the swarm
num_particles = 10
dimensions = 3
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Create a Particle Swarm Optimizer
optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=dimensions, options=options)

# Run the optimization
best_position, _ = optimizer.optimize(objective_function, iters=100)

# Display the result
print("Best Position:", best_position)
print("Objective Value at Best Position:", objective_function(best_position))
