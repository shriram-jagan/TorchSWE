# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Notebook to postprocess the pickle file. 
# The pickle file contains the solution variables and the grid information. 
# This is usually used to check results and/or compare different runs/backends
# -

import pickle as pkl 
import numpy
import pathlib

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import numpy as np


# +
def read(case, filename: str):
    return pkl.load(open(case.joinpath(filename), "rb"))

def get_solution_variables(case, filename: str):
    s = read(case, filename)
    w = s['w']
    h = s['h']
    hu = s['hu']
    x = s['x'][:-1] # x has one extra point; fix it in pkl dump
    dx = s['dx']

    return x, dx, w, h, hu


# -

def plotter(data, filename: str = None):
    variables = ['h', 'u']

    nrows = len(variables)
    ncols = 1
    nplots = int(nrows * ncols)

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(12,9))
    for row in range(nrows):
        ax = axes[row]

        variable = variables[row]

        h = ax.contourf(data[variable])
        plt.colorbar(h, ax=ax)

        ax.set_xlabel('$N_{x}$')
        ax.set_ylabel('$N_{y}$')
        ax.set_title("{}".format(variable))

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


# #### INPUT: Path to the directory with the pickle file

case_vec   = pathlib.Path("<dir>/vectorize").expanduser().resolve()
case_novec = pathlib.Path("<dir>/no_vectorize").expanduser().resolve()

# +
filename = 'end.pkl'

vec = read(case_vec, filename)
no_vec = read(case_novec, filename)

# +
###### relevant quantities: hu, h, u, w (hv and v will be close to zero, so not that interesting)
# -

quantity = 'hu'  

# ### Plot one quantity and compare vectorized vs array-based implementations

# +
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,6))

ax = axes[0]
h = ax.contourf(vec[quantity])
plt.colorbar(h, ax=ax)
ax.set_xlabel('$N_x$')
ax.set_ylabel('$N_y$')
ax.set_title('Vectorized: ' + quantity)

ax = axes[1]
h = ax.contourf(no_vec[quantity])
plt.colorbar(h, ax=ax)
ax.set_xlabel('$N_x$')
ax.set_ylabel('$N_y$')
ax.set_title('Array-based: ' + quantity)

plt.tight_layout()

# -

# ### Plot all quantities

# +
# end  = read(case_vec, filename)
# plotter(end)
# -
# ### Compare the two versions for different quantities and compute the relative error 


relative_error = {quantity: np.linalg.norm(vec[quantity] - no_vec[quantity])/np.linalg.norm(no_vec[quantity]) for quantity in ['hu', 'h', 'w']}
print(relative_error)


