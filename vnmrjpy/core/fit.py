import numpy as np
from lmfit import minimize, Parameters
import lmfit
import matplotlib.pyplot as plt
import itertools
import vnmrjpy as vj

#TODO abysmal performace. make the result 3d wrapper properly, and
# everything else along with it...
"""
Collection of functions aiding the fitting of 3D timeseries data.
lmfit is the main python module utilized. Most functions here are wrappers
for lmfit class methods to help deal with independent fitting on 3d dataseries
"""

#TODO eval_uncertainty wrapper

#--------------------Wrappers for ModelResult.attributes-----------------------

def get_best_values3d(res3d, model):
    """Extract fit3d result best_value attribute into a list of 3d volumes

    Args:
        res3d -- 3d numpy array of model.ModelResult; output of fit3d
        model -- lmfit Model of the fit
    Return:
        attr3d_list -- list of numpy arrays in same order as model.param_names
    """

    param_names = model.param_names
    
    data_type = [type(i) for i in list(res3d[0,0,0].best_values.values())]
    num = len(param_names)
    attr3d_list = []
    shape = res3d.shape
    # creating empty arrays
    for i in range(num):
        arr = np.zeros(shape,dtype=data_type[i])
        attr3d_list.append(arr)
    # fill the arrays
    for x, y, z in itertools.product(*map(range, shape)):
        vals = list(res3d[x,y,z].best_values.values())
        for k, attr3d in enumerate(attr3d_list):
            attr3d[x,y,z] = vals[k]
            
    return attr3d_list

def get_best_fit3d(res3d, time_dim):
    """Extract fit3d result best_fit attribute into a 3d volume of timeseries

    This is the fitted model evaluated on the input grid

    Args:
        res3d -- 3d numpy array of model.ModelResult; output of fit3d
        time_dim -- length of time dimension
    Return:
        attr3d -- 4d numpy array of best fit data dims: [x,y,z,time]
    """

    # create empty array 
    data_type = type(res3d[0,0,0].best_fit[0])
    shape = list(res3d.shape) + [time_dim]
    attr3d = np.zeros(shape,dtype=data_type)
    # fill the arrays
    for x, y, z in itertools.product(*map(range, shape[0:3])):
        attr3d[x,y,z,:] = res3d[x,y,z].best_fit

    return attr3d

def get_chisqr3d(res3d):
    """Extract fit3d result chisqr attribute into a 3d volume

    Args:
        res3d -- 3d numpy array of model.ModelResult; output of fit3d
    Return:
        attr3d -- numpy arrays of chi-square statistics of fit
    """

    # create empty array 
    data_type = type(res3d[0,0,0].chisqr)
    shape = res3d.shape
    attr3d = np.zeros(shape,dtype=data_type)
    # fill the arrays
    for x, y, z in itertools.product(*map(range, shape)):
        attr3d[x,y,z] = res3d[x,y,z].chisqr

    return attr3d

#-----------------------Wrapper for Model.make_params method-------------------

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

#---------------------------Wrapper for Model.fit()----------------------------
#TODO this is very inefficient....
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

        raise(Exception('SGE support not implemented'))
