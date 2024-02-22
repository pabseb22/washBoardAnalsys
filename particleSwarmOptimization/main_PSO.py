import numpy as np
import pyswarms as ps
from cellbedform_PSO import CellBedform
from scipy.optimize import minimize

# Problemas: 
# Contra que comparo
# Como el optimizer puede encontrar algo si la funcion me devuelve por cada step un corte en x, verifica cada step?
# Que corte elegimos, analiza todos?
# En caso de que pueda optimizar algo, como podemos replicar sus resultados si se genera una superficie aleatoria. Debo dejar seteada una misma superficie?
# Que parametros deberia optimizar?
# Cuantos steps seran suficientes?

D = 0.8
Q = 0.6

dx = 150
dy = 40
y_cut = 20
steps = 101

def wave1(x, a, b, c):
    return a * np.sin(b * x + c)

# Objective function to minimize (quadratic function)
def objective_function(params):
    L0_, b_ = params
    cb = CellBedform(grid=(dx, dy), D=D, Q=Q, L0=L0_, b=b_, y_cut=y_cut)
    y_cuts, initial_h = cb.run(steps)
    diff = y_cuts[100] - wave1(2,1,3,1)
    return np.sum(diff**2)

# Initial guess for the parameters
params_initial = [7.3, 2]

# Call the optimizer
result = minimize(objective_function, params_initial, method='Nelder-Mead')

# The optimal parameters are stored in result.x
print(result)
print(result.x)

# Initialize the swarm
# num_particles = 10
# dimensions = 3
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# # Create a Particle Swarm Optimizer
# optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=dimensions, options=options)

# # Run the optimization
# best_position, _ = optimizer.optimize(objective_function, iters=100)

# # Display the result
# print("Best Position:", best_position)
# print("Objective Value at Best Position:", objective_function(best_position))


