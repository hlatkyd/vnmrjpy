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
    
    # setting data: 90deg rotation is done by swapping and flipping
    oldaxes = [i for i in range(varr.data.ndim)]
    newaxes = copy.copy(oldaxes)
    newaxes[0:3] = swapaxes
    #varr.data = np.moveaxis(varr.data, newaxes, oldaxes) 
    varr.data = np.moveaxis(varr.data, oldaxes, newaxes) 
    
    # ------flipping as part of rotation----------------------
    varr.data = _flip_axes(varr.data,flipaxes)

    # setting sdims
    #varr = _move_sdims(varr,new_sdims) 
    sdims_partial = varr.sdims[:3]
    sdims_kept = varr.sdims[3:]
    new_sdims_partial = [i[1] for i in sorted(zip(swapaxes, sdims_partial))]
    
    new_sdims = new_sdims_partial+sdims_kept

    varr.space = 'anatomical'
    varr.set_dims()
    varr.sdims = new_sdims
    # -----additional custom flip data on axes------------------

    #this is 
    #varr.flip_axis('z')
    #varr.flip_axis('phase')
    #varr.flip_axis('slice')
    
    # 3D should be flipped. Dunno why, but seems to work
    if varr.pd['apptype'] in ['im3D','im3Dfse']:
        varr.flip_axis('slice')

    # update header
    varr.set_nifti_header()

    return varr

def _to_global(varr):
    raise(Exception('Not implemented'))
    return varr

def _to_local(varr):
    """Transform to [phase, read, slice, etc..]

    Sets appropriate varray attributes including nifti_header
    If it is first called from to_kspace() automatically, and also from other
    'read' methods
    """
    #TODO autocall from other read functions
    # if space is None, then the call is from to_kspace()
    if varr.space == None:
    
        varr.space = 'local'
        varr.sdims = ['phase','read','slice','time','rcvr']
        varr.set_dims()
        if vj.config['swap_x_grad'] == True:
            varr.flip_axis('x')
        varr.set_nifti_header()

    elif varr.space == 'anatomical':
        raise(Exception('not implemented yet'))
    elif varr.space == 'local':
        # do nothing, already in local space
        return varr 

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

def _anatomical_swaps(pd):
    """Return swap and flip arrays for data transform to anatomical

    use_hardcoded: no-brain implementation for 90deg rots
    """
    use_hardcoded = True
    # hardcoded for 90degs
    if use_hardcoded:
        if _check90deg(pd) != True:
            raise(Exception('Not implemented'))

        ori = pd['orient']
        if ori == 'trans':
            #swap = [0,1,2]
            swap = [0,2,1]
            flip = [1,2]
        elif ori == 'trans90':
            #swap = [1,0,2]
            swap = [2,0,1]
            flip = [0]
        elif ori == 'sag':
            #swap = [1,2,0]
            swap = [2,1,0]
            flip = [1,2]
        elif ori == 'sag90':
            #swap = [2,1,0]
            swap = [1,2,0]
            flip = [0]
        elif ori == 'cor':
            #swap = [0,2,1]
            swap = [0,1,2]
            flip = [1]
        elif ori == 'cor90':
            swap = [1,0,2]
            flip = []

        return swap, flip
    # with rot matrix
    else:
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


def _set_dims(varr):
    """Set varray dims"""
    if varr.space == 'local' or varr.space == None:
        if varr.dims != None:
            return varr
        if _check90deg(varr.pd) != True: 
            varr.dims = 'oblique'
            return varr
        swapax, flipax = _anatomical_swaps(varr.pd)
        swap = swapax
        anat_dims = ['x','y','z']
        dims = [i[1] for i in sorted(zip(swap,anat_dims))]
        varr.dims = dims
        return varr

    elif varr.space == 'anatomical':
        # TODO set dims only if 90 deg rot case
        if _check90deg(varr.pd): 
            varr.dims = ['x','y','z']
        else:
            varr.dims = 'oblique'
        return varr

def _flip_axis(varr,axis):
    """Flip data on axis 'x','y','z' or 'phase','read','slice'"""
    if axis in varr.dims:
        ax = varr.dims.index(axis)
        varr.data = np.flip(varr.data,axis=ax)
    elif axis in varr.sdims:
        ax = varr.sdims.index(axis)
        varr.data = np.flip(varr.data,axis=ax)
    else:
        raise(Exception('Axis not specified correctly'))
    
    return varr
