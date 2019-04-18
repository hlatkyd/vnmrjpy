import numpy as np
from lmfit import minimize, Parameters
import matplotlib.pyplot as plt
import itertools
import vnmrjpy as vj

"""
Collection of functions aiding the fitting of 3D timeseries data
"""

def residual1D(params, model, x, data, eps_data):
    """Return residual for least squares fitting"""
    return (data-model) / eps_data

def minimize3D(residual, params3D, args3D, mask3D):
    """Wrapper for lmfit.minimize for 3D timeseries

    Intended behavior is the same, except inputs and outputs are in
    numpy arrays.

    Args:
        residual -- residual function for 1D data
        params3D -- lmfit.Parameters in 3D numpy array
        args3D -- (x,data, ydata, eps_data) in 3D array
        mask3D -- 3D np.array of same spatial shape as data
    Return:
        out3D
    """
    (x_dim, y_dim, z_dim) = args3D[1].shape[0:3]
    out3D = np.zeros((x_dim,y_dim, z_dim))
    for x, y, z in itertools.product(*map(range, (x_dim,y_dim,z_dim))):
        if mask3D[x,y,z] == 0:
            zero = minimize.MinimizerResults()
            out3D[x,y,z] = 1        
            continue
        else:
            (xdata,ydata,eps_data) = args3D[x,y,z]
            params = params3D[x,y,z]
            try:
                out3D[x,y,z] =\
                 minimize(residual, params, args=(xdata,ydata, eps_data))
            except:
                pass
    return out3D

def fit3D_leastsq_2stage():

    pass

def reinit3D():

    pass

