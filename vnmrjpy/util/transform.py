import vnmrjpy as vj
import numpy as np
from math import sin, cos

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

    if check_90deg(procpar) == False:
        raise(Exception('Only supported with rotations of multiples of 90deg'))

    p = vj.io.ProcparReader(procpar).read()
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
    R = np.array(R)
    Inv = np.linalg.inv(R)
    
    # or inv R?
    axes_transform_matrix = np.linalg.inv(R)
    print(axes_transform_matrix)

    if len(data.shape) == 4:
        orig = np.array([0,1,2,3])+1
        new = np.append(axes_transform_matrix.dot(orig[:-1]),orig[-1])
        new = np.absolute(new)
        new = np.array(new,dtype=int)-1
        print(new)
        orig = orig -1
        data = np.moveaxis(data, orig, new)
        return data

    elif len(data.shape) == 5: 
        orig = np.array([0,1,2,3,4])+1
        new = np.append(axes_transform_matrix.dot(orig[1:-1]),orig[-1])
        new = np.append(orig[0],new)
        new = np.absolute(new)
        new = np.array(new,dtype=int)-1
        orig = orig -1
        data = np.moveaxis(data, orig, new)
        return data

    else:
        raise(Exception('not implemented for dimensions other than 4 or 5'))

    #TODO transform affine as well
    


