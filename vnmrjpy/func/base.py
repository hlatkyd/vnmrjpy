import vnmrjpy as vj
import numpy as np
from vnmrjpy.core.utils import vprint
import copy
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt

"""
Basic functions on varrays.
Includes:
    noise_mask
    average
    concatenate 
"""

def mask(varr, threshold=None, mask_out=True):
    """Create mask on image: set voxel data to 0 where image is below threshold.

    Noise threshold is assumed if set to None

    """
    def _vertical_project(data):

        data = np.mean(data,axis=(0,1))
        return data    

    def _guess_threshold(vertical_projection):

        # TODO account for noise variance in a more normal way...
        thresh = 2 * np.min(vertical_projection)
        return thresh

    # guessing a threshold
    if threshold == None: 
        vprint('Making mask, assuming noise to be in the'+
                'top voxels when determining threshold')
        orig_space = varr.space
        varr.to_anatomical()
        magnitude = vj.core.recon.ssos(varr.data)  # simple combine rcvrs
        # top down is dim2 backwards
        vert_proj = _vertical_project(magnitude)
        vert_min = np.min(vert_proj)
        vert_max = np.max(vert_proj)

        threshold = _guess_threshold(vert_proj)

    # masking
    mask = np.zeros_like(varr.data,dtype='float32')
    mask[np.abs(varr.data) > threshold] = 1.0

    # filling mistakenly masked inner voxels:
    mask = median_filter(mask, size=3)
    # 2nd pass
    mask = median_filter(mask, size=3)

    varr.data = varr.data * mask

    # transform back to original space for consistency
    if orig_space == 'anatomical':
        pass
    elif orig_space == 'local':
        varr.to_local()
    elif orig_space == 'global':  # TODO this is not ready though
        varr.to_global()

    # return results
    if mask_out:
        return varr, mask
    else:
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

