import os
import sys

os.chdir("/work4/clf/david/Bayesian/BayesianGit/BayesianOptimiser2")

# Add the new working directory to sys.path
sys.path.append(os.getcwd())

import numpy as np
import scipy.constants as con
import matplotlib.pyplot as plt
import BOPackage


"""
Define the objective function.
"""
def BraninHoo(X):

    x1 = X[:, 0]
    x2 = X[:, 1]
    
    # Calculate the function value
    y = (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + (5 / np.pi) * x1 - 6) ** 2
    y += 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10

    # # Add the constraint
    # constraint = (x1 - 2.5) ** 2 + (x2 - 7.5) ** 2 - 50 >= 0
    # y[constraint] = 200  # Apply penalty for constraint violation
        
    # Make function negative to find the minimum
    y = -y.reshape(-1,1)

    return y


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Example data
X1 = np.linspace(-5, 10, 100)
X2 = np.linspace(0, 15, 100)
X1_grid, X2_grid = np.meshgrid(X1, X2)
X = np.c_[X1_grid.ravel(), X2_grid.ravel()]
y = BraninHoo(X)

# Fit a Gaussian Process with an RBF kernel
kernel = RBF(length_scale=1.0)  # You can start with an initial guess
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, y)

# Get the optimized length scale
length_scale = gp.kernel_.length_scale
print(f"Estimated Length Scale: {length_scale / 2}")
