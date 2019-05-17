import vnmrjpy as vj
import numpy as np
import copy

"""
Basic functions on varrays.
Includes:
    noise_mask
    average
    concatenate 
"""

def _check_varrlist_integrity(vlist):
    """Return true if shapes and datatypes are the same"""
    shape = vlist[0].data.shape
    datatype = vlist[0].data.dtype
    for v in vlist:
        if v.data.shape != shape:
            raise(Exception("Data shapes don't match"))
        if v.data.dtype != datatype:
            raise(Exception("Data types don't match"))
    return True

def mask(varr, threshold=None):
    """Create mask on image: set voxel data to 0 where image is below threshold.

    Noise threshold is assumed if set to None

    """ 
    return varr

def average(varr_list,method='default', dim='time'):
    """Return time averaged vj.varray"""
    
    if _check_varrlist_integrity(varr_list) == True:
        pass
    if dim == 'time':
        axis = 3
    data_list = [v.data for v in varr_list]
    newdata = np.mean(data_list,axis=axis)
    newvarr = varr_list[0]
    newvarr.data = newdata
    return newvarr

def concatenate(varr_list, params=['te'], dim='time'):
    """Concatenate data of multiple varray along a dimension

    Args:
        varr_list
        params -- list of procpar parameters to concatenate as well, in order.
    Return:
        varr -- new instance of a varray, with now concatenated data and
                modified pd
    """
    if dim =='time':
        concat_axis = 3
    # concatenate data
    data_list = [varr.data for varr in varr_list]
    newdata = np.concatenate(data_list, axis=concat_axis)
    # creating output varray from first in list
    outvarr = varr_list[0]
    outvarr.data = newdata

    newpd = varr_list[0].pd
    # make pd dict elements
    for k, par in enumerate(params):
        vals = []
        for n in range(len(varr_list)):
            vals.append(varr_list[n].pd[par])
        partial_pd = {par:vals}
        newpd.update(partial_pd)
 
    outvarr.pd = newpd
    return outvarr
    
