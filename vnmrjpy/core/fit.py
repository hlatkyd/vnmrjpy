import numpy as np
from lmfit import minimize, Parameters
import lmfit
import matplotlib.pyplot as plt
import itertools
import vnmrjpy as vj

"""
Collection of functions aiding the fitting of 3D timeseries data.
lmfit is the main python module utilized.
"""
#TODO
# make these more to the point
def get_params3d(res3d, model, attribute='best_values'):
    """Extract fit3d result into a list of 3d volumes of parameters

    Args:
        res3d -- 3d numpy array of model.ModelResult; output of fit3d
        model -- lmfit Model of the fit
    Return:
        param3d_list -- 
    """

def make_params3d(model, shape3d, **kwargs):
    """Wrapper for lmfit.Model.make_params() . Return 3d numpy array of Parameters
    
    Args:
        model -- lmfit.Model instance
        shape3d -- tuple of lenght 3
        **kwargs -- **kwargs to put into model.make_params() method 
    Return:
        params3d -- 3d numpy array filled with lmfit Parameters instances
    """
    param1d = model.make_params(**kwargs)
    data_type= type(param1d)
    params3d = np.empty(shape=shape3d,dtype=data_type)
    params3d.fill(param1d)
    return params3d

def fit3d(func, y3d, params3d, x3d, method='individual',device='cpu', **kwargs):
    """Perform model fitting on 3D timeseries data with lmfit Model.fit() method

    For each point in the 3d space an lmfit.Model is created and Model.fit is
    performed on the timeseries at that point. The model is created 

    Args:
        func -- callable python function of form f(x, *args, **kwargs)
        y3d -- 4d numpy.ndarray of y data series (measurement, dims=[x,y,z,val])
        params3d -- initialized lmfit.Parameters in 3d numpy array
        x3d -- 4d numpy.ndarray of x data series (grid, dims=[x,y,z,val])
        **kwargs -- **kwargs going into model.fit() method
    Return:
        res3d -- lmfit.model.ModelResult in 3d numpy array
    """

    (x_dim, y_dim, z_dim) = y3d.shape[0:3]
    res3d = np.empty((x_dim,y_dim,z_dim),dtype=type(lmfit.model.ModelResult))
    model = lmfit.Model(func)
    if device == 'cpu':
        for x, y, z in itertools.product(*map(range, (x_dim,y_dim,z_dim))):
            res = model.fit(y3d[x,y,z], params3d[x,y,z], x=x3d[x,y,z], **kwargs)
            res3d[x,y,z] = res
        return res3d

    elif device == 'sge':
        raise(Exception('SGE not supported yet'))

#TODO
# deprecated
def residual1D(params, model, x_data, y_data, eps_data):
    """Return residual for least squares fitting
    
    Args:
        params -- lmfit.Parameters
        model -- callable function of model
        x_data -- x axis data (as a grid)
        y_data -- y axis data (as measured on grid)
        eps_data -- y_data uncertainty
    Return:
        residual
    """
    model_fit = model(x_data, *par_list)
    residual =  (y_data - model_fit) / eps_data
    return residual

#TODO
#deprecated
def minimize3D(residual, params3D, args3D, mask3D=None):
    """Wrapper for lmfit.minimize for 3D timeseries

    Intended behavior is the same, except inputs and outputs are in
    numpy arrays.

    Args:
        residual -- residual function for 1D data, callable
        params3D -- lmfit.Parameters in 3D numpy array
        args3D -- (x,data, ydata, eps_data) in 3D array
        mask3D -- 3D np.array of same spatial shape as data
    Return:
        out3D -- lmfit.minimize output (MinimizerResults)
                    in 3D numpy array
    """
    (x_dim, y_dim, z_dim) = args3D[1].shape[0:3]
    out3D = np.zeros((x_dim,y_dim, z_dim))
    #TODO make zeroed results to same format as nonzero
    zero_out = 0
    zero = minimize.MinimizerResults()

    for x, y, z in itertools.product(*map(range, (x_dim,y_dim,z_dim))):
        if mask3D[x,y,z] == 0:
            out3D[x,y,z] = zero
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
#TODO
# deprecated
def fit3D_leastsq_2stage():

    pass

#TODO
# deprecated
def reinit3D():

    pass

