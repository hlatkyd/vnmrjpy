import vnmrjpy as vj
import numpy as np

"""
Collection of helper functions for epi and epip k-space formation
and preprocessing.
"""

def _get_navigator_echo(kspace,pd,phase_dim=1):
    """Return navigator echo from full received data"""
    if pd['navigator'] == 'y':
        navigator = kspace[:,0:1,:,:,:]
        return navigator
    else:
        return None

def _prepare_shape(kspace,pd,phase_dim=1):
    """Remove unused and navigator echo from kspace data"""
    if pd['navigator'] == 'y':
        kspace = kspace[:,2:,:,:,:]
    else:
        kspace = kspace[:,1:,:,:,:]
    return kspace

def _kzero_shift(kspace,p,phase_dim=1):
    """Shift data so first echo is at the proper space according to kzero"""

    kzero = int(p['kzero'])
    kspace = np.roll(kspace,kzero,axis=phase_dim)
    return kspace

def _reverse_even(kspace, phase_dim=1):
    """Reverse even echoes"""
    odd = kspace[:,0::2,...]
    oddflip = np.flip(odd,axis=2)
    kspace[:,0::2,...] = oddflip
    return kspace

def _reverse_odd():
    pass
