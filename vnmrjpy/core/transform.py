import vnmrjpy as vj
import numpy as np

"""
Contains functions to transform vnmrjpy.varray into other coordinate systems.
These functions are not intented to be called on their own, but from the
appropriate method of a varray object. More documentation is present at
vnmrjpy/core/varray.py
"""
def _to_scanner(varr):

    pass

def _to_anatomical(varr):

    pass

def _to_global(varr):

    pass

def _to_local(varr):

    pass

def _check90deg(pd):
    """Return True if the Euler angles are 0,90,180, etc"""
    psi, phi, theta = int(p['psi']), int(p['phi']), int(p['theta'])
    if psi % 90 == 0 and phi % 90 == 0 and theta % 90 == 0:
        return True
    else:
        return False

def _flipaxes(varr,axes):
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

