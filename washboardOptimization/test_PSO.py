
# import modules
import numpy as np

# create a parameterized version of the classic Rosenbrock unconstrained optimzation function
def rosenbrock_with_args(x, a, b, c=0):
    print("TEST" , a)
    f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
    print(f)
    return f

from pyswarms.single.global_best import GlobalBestPSO

# instatiate the optimizer
x_max = 10 * np.ones(2)
x_min = -1 * x_max
print(x_max)
print(type(x_max))
# bounds = (x_min, x_max)
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# optimizer = GlobalBestPSO(n_particles=5, dimensions=2, options=options, bounds=bounds)

# # now run the optimization, pass a=1 and b=100 as a tuple assigned to args

# kwargs={"a": 9.0, "b": 100.0, 'c':0}
# cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, **kwargs)