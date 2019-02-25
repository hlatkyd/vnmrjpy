import vnmrjpy as vj
import numpy as np
from math import sin, cos

def make_cubic_affine(data):
    """Make Nifti affine assuming imaging volume is a cube

    Args:
        data (np.ndarray) -- 3D or 4D input data to make affine for
    Return:
        affine (np.ndarray) -- 4x4 or 3x3? diagonal matrix

    Useful for easy viewing, but don!t use it for serios work.
    """
    x = 1/data.shape[0]
    y = 1/data.shape[1]
    z = 1/data.shape[2]
    dim_arr = np.array([x,y,z,1])
    affine = np.eye(4)*dim_arr
    return affine

def make_scanner_affine(procpar):
    """Make appropriate global affine for Nifti header.

    This is needed for viewing and saving nifti in scanner space
    """ 
    p = vj.io.ProcparReader(procpar).read()
    orient = p['orient']

def check_90deg(procpar):
    """Make sure all rotations are multiples of 90deg

    Euler angles of rotations are psi, phi, theta,

    Args:
        procpar
    Return:
        True or False

    """
    p = vj.io.ProcparReader(procpar).read()
    psi, phi, theta = int(p['psi']), int(p['phi']), int(p['theta'])
    if psi % 90 == 0 and phi % 90 == 0 and theta % 90 == 0:
        return True
    else:
        return False

def to_scanner_space(data, procpar):
    """Transform data to scanner coordinate space by properly swapping axes

    Standard vnmrj orientation - meaning rotations are 0,0,0 - is axial, with
    x,y,z axes (as global gradient axes) corresponding to phase, readout, slice.
    vnmrjpy defaults to handling numpy arrays of:
                (receivers, phase, readout, slice/phase2, time*echo)
    but arrays of
                        (receivers, x,y,z, time*echo)
    is also desirable in some cases (for example registration in FSL flirt)

    Euler angles of rotations are psi, phi, theta,
    Args:
        data (3,4, or 5D np ndarray) -- input data to transform
        procpar (path/to/file) -- Varian procpar file of data

    Return:
        swapped_data (np.ndarray)
    """

    def _orthotransform(data,orient):
        """Simple swaping of axes in case of orthogonal scanner-image axes

        This was done to avoid thinking. Would benefit from rewriting.
        """
        dims = len(data.shape)
        
        if orient == 'trans':
            pass
        if orient == 'trans90':
            if dims == 3:
                data = np.swapaxes(data,0,1)
                data = np.flip(data,axis=1)
            elif dims == 4:
                data = np.swapaxes(data,0,1)
                data = np.flip(data,axis=1)
            elif dims == 5:
                data = np.swapaxes(data,1,2)
                data = np.flip(data,axis=2)
        if orient == 'sag':
            if dims == 3:
                data = np.swapaxes(data,0,1)
                data = np.swapaxes(data,0,2)
            elif dims == 4:
                data = np.swapaxes(data,0,1)
                data = np.swapaxes(data,0,2)
            elif dims == 5:
                data = np.swapaxes(data,1,2)
                data = np.swapaxes(data,1,3)
        if orient == 'sag90':
            if dims == 3:
                data = np.swapaxes(data,0,2)
            elif dims == 4:
                data = np.swapaxes(data,0,2)
            elif dims == 5:
                data = np.swapaxes(data,1,3)
        if orient == 'cor':
            if dims == 3:
                data = np.swapaxes(data,1,2)
                data = np.flip(data,axis=1)
            elif dims == 4:
                data = np.swapaxes(data,1,2)
                data = np.flip(data,axis=1)
            elif dims == 5:
                data = np.swapaxes(data,2,3)
                data = np.flip(data,axis=2)
        if orient == 'cor90':
            if dims == 3:
                data = np.swapaxes(data,0,1)
                data = np.swapaxes(data,1,2)
            elif dims == 4:
                data = np.swapaxes(data,0,1)
                data = np.swapaxes(data,1,2)
            elif dims == 5:
                data = np.swapaxes(data,1,2)
                data = np.swapaxes(data,2,3)
        
        # correct in case X gradient is inverted
        if vj.config['swap_x_grad']:
            if dims == 3 or dims == 4:
                data = np.flip(data,axis=0)
            elif dims == 5:
                data = np.flip(data,axis=1)
        
        return data
                

    #TODO this is for later, for more sophisticated uses perhaps
    def _calc_matrix(p):
        """calculate rotation matrix"""

        psi = int(p['psi'])/360*(2*np.pi)
        phi = int(p['phi'])/360*(2*np.pi)
        theta = int(p['theta'])/360*(2*np.pi)
        # rotation matrix with the euler angles
        R = [[cos(phi)*cos(psi)-sin(phi)*cos(theta)*sin(psi),\
                    -sin(phi)*cos(psi)-sin(psi)*cos(theta)*cos(phi),\
                    sin(theta)*sin(psi)],\
            [cos(phi)*sin(psi)+sin(phi)*cos(theta)*cos(psi),\
                    -sin(phi)*sin(psi)+cos(psi)*cos(theta)*cos(phi),\
                    -sin(theta)*cos(psi)],\
            [sin(theta)*sin(phi), sin(theta)*cos(phi), cos(theta)]]
    #-------------------------------main--------------------------------
    if len(data.shape) == 4 or len(data.shape) == 5:
        pass
    else:
        raise(Exception('Only 3d, 4d or 5d data allowed'))
    p = vj.io.ProcparReader(procpar).read()

    if check_90deg(procpar) == True:
        
        return _orthotransform(data, p['orient'])
    
    else:
        raise(Exception('Only supported with rotations of multiples of 90deg'))

    #TODO transform affine as well
    


