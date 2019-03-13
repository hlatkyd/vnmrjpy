import vnmrjpy as vj
import numpy as np
import copy

"""
Contains functions to transform vnmrjpy.varray into other coordinate systems.
These functions are not intented to be called on their own, but from the
appropriate method of a varray object. More documentation is present at
vnmrjpy/core/varray.py
"""
def _to_scanner(varr):
    """Transform data to scanner coordinate space by properly swapping axes

    Standard vnmrj orientation - meaning rotations are 0,0,0 - is axial, with
    x,y,z axes (as global gradient axes) corresponding to phase, readout, slice.
    vnmrjpy defaults to handling numpy arrays of:
                (receivers, phase, readout, slice/phase2, time*echo)
    but arrays of
                        (receivers, x,y,z, time*echo)
    is also desirable in some cases (for example registration in FSL flirt)

    Euler angles of rotations are psi, phi, theta,

    Also corrects reversed X gradient and sliceorder

    Args:
        data (3,4, or 5D np ndarray) -- input data to transform
        procpar (path/to/file) -- Varian procpar file of data

    Return:
        swapped_data (np.ndarray)
    """
    return varr

def _to_anatomical(varr):

    if varr.space == 'anatomical':
        return varr
    if _check90deg(varr.pd) == False:
        raise(Exception('Only multiples of 90 deg allowed here.'))    
    # TODO this is the old method
    #new_sdims, flipaxes = _anatomical_sdims(varr.pd['orient'])

    # better method is via rotation matrix
    swapaxes, flipaxes = _anatomical_swaps(varr.pd)
    
    # setting data
    oldaxes = [i for i in range(varr.data.ndim)]
    newaxes = copy.copy(oldaxes)
    newaxes[0:3] = swapaxes
    varr.data = np.moveaxis(varr.data, newaxes, oldaxes) 
    
    #varr.data = np.moveaxis(varr.data, oldaxes, newaxes)
    varr.data = _flip_axes(varr.data,flipaxes)

    # setting sdims
    #varr = _move_sdims(varr,new_sdims) 
    sdims_partial = varr.sdims[:3]
    sdims_kept = varr.sdims[3:]
    new_sdims_partial = [i[1] for i in sorted(zip(swapaxes, sdims_partial))]
    
    new_sdims = new_sdims_partial+sdims_kept

    varr.sdims = new_sdims
    #varr.set_nifti_header()
    varr.space = 'anatomical'

    return varr

def _to_global(varr):

    return varr

def _to_local(varr):
    """Transform back to [phase, read, slice, etc..]"""
    return varr

def _check90deg(pd):
    """Return True if the Euler angles are 0,90,180, etc"""
    psi, phi, theta = int(pd['psi']), int(pd['phi']), int(pd['theta'])
    if psi % 90 == 0 and phi % 90 == 0 and theta % 90 == 0:
        return True
    else:
        return False

def _flip_axes(data,axes):
    """Return flipped data on axes"""
    for ax in axes:
        data = np.flip(data, axis=ax)
    return data

# TODO consider delete
def _flip_axes_deprecated(varr,axes):
    """Flip varr.data on multiple axes.

    Args:
        axes -- Can be a list of 'read','phase','slice', or 'x','y','z'
    """
    axnum_list = []
    try:
        for ax in axes:
            axnum = varr.dims.index(ax)
            axnum_list.append(axnum)
    except:
        axnum = varr.sdims.index(ax)
        axnum_list.append(axnum)
    for ax in axnum_list:
        varr.data = np.flip(varr.data,axis=ax)
    return varr

def _anatomical_swaps(pd):
    """Return sdims for anatomical oriantation"""

    rot_matrix = vj.core.niftitools._qform_rot_matrix(pd)
    inv = np.linalg.inv(rot_matrix).astype(int)
    swap = inv.dot(np.array([1,2,3], dtype=int))
    flipaxes = []
    for num, i in enumerate(swap):
        if i < 0:
            flipaxes.append(num)
    swapaxes = (np.abs(swap) - 1).astype(int)

    return swapaxes, flipaxes
    

def _move_sdims(varr,sdims_new):
    """Move varray data aexes according to new sdims"""

    # move

    # set new sdims
    varr.sdims = sdims_new
    return varr
