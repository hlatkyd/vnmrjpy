import vnmrjpy as vj
import numpy as np
from math import sin, cos

def _get_swap_array(orient):
    """Gives how to reorder and flip the axes to fit scanner space

    [phase, readout, slice] -> [x,y,z]
    
    Used with np.moveaxes

    Args:
        orient -- procpar parameter 'orient'
    Returns:
        arr -- sequence of new axes
        flipaxis
    """
    if orient == 'trans':
        arr = [0,1,2]
        flipaxis = None
    elif orient == 'trans90':
        arr = [1,0,2]
        flipaxis = 1
    elif orient == 'sag':
        arr = [2,0,1]
    elif orient == 'sag90': 
        arr = [2,1,0]
        flipaxis = None
    elif orient == 'cor':
        arr = [0,2,1]
        flipaxis = 1
    elif orient == 'cor90': 
        arr = [1,2,0]
        flipaxis = None
    else:
        raise(Exception('Other orientations not implemented'))

    return arr, flipaxis

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
    Prabably safer to do it from scratch than transforming...
    Translations are accounted for. Gaps are added to thinckness
    
    Args:
        data (np.ndarra) -- input data, 4D or 5D
        procpar 
    Return:
        affine (np.ndarray) -- 4x4 diagonal matrix
    """
    p = vj.io.ProcparReader(procpar).read()
    orient = p['orient']
    # epip and other have different parameter names on phase and readout matrix
    if 'epip' not in p['pslabel']:
        phase = int(p['np'])//2
        read = int(p['nv'])
        # these are in cm!
        lro = float(p['lro'])*10
        lpe = float(p['lpe'])*10
        if '2D' in p['apptype']:
            slices = int(p['ns'])
            thk = float(p['thk'])
            gap = float(p['gap'])
            arr = [lpe/phase,lro/read,(thk+gap)]
        elif '3D' in p['apptype']:
            phase2 = int(p['nv2'])
            lpe2 = float(p['lpe2'])*10
            arr = [lpe/phase,lro/read,lpe2/phase2]
        else:
            raise(Exception)
            
    elif 'epip' in p['pslabel']:
        pass
    else:
        raise(Exception)

    swaparr, flipaxis = _get_swap_array(p['orient'])
    sortedarr = sorted(zip(swaparr,arr))
    arr = [sortedarr[i][1] for i in range(3)]+[1.0]
    affine = np.array(np.eye(4,dtype=float)*arr,dtype=float)
    return affine
    
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
        newarr, flipaxis = _get_swap_array(orient)  
        if dims == 3:
            data = np.moveaxis(data, [0,1,2], newarr)
        elif dims == 4:
            newarr = newarr+[3]
            data = np.moveaxis(data, [0,1,2,3], newarr)
        elif dims == 5:
            if flipaxis != None:
                flipaxis = flipaxis+1
            newarr = [0]+newarr+[4]
            data = np.moveaxis(data, [0,1,2,3,4], newarr)

        if flipaxis != None:
            data = np.flip(data, axis=flipaxis) 

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
    


