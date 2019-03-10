import vnmrjpy as vj
import numpy as np
import nibabel as nib
import warnings
from math import sin, cos

def _set_nifti_header(varr):
    """Set varray.nifti_header attribute for the current space"""
    # init empty header
    header = nib.nifti1.Nifti1Header()

    if varr.space == 'scanner':

        pass
        # use sform

    
    elif varr.space == 'local':

        # use qform only
        dim_count = len(varr.data.shape)
        header['dim'][0] = dim_count
        for i in range(dim_count):
            header['dim'][i+1] = varr.data.shape[i]
        # setting pixdom based on procpar data
        d = _get_pixdims(varr.pd)
        
        # see qfac in doc
        qfac = 1
        header['pixdim'][0] = qfac
        for i in range(3):
            header['pixdim'][i+1] = d[i]

        # set local dim info
        header.set_dim_info(freq=0,phase=1,slice=2)

        # setting qform
        header['qform_code'] = 1
        q_affine = _qform_affine(varr)
        
        header.set_qform(q_affine,code=1)

    # setting datatype 
    if varr.vdtype == 'complex64': 
        header['datatype'] = 32
    if varr.vdtype == 'float32': 
        header['datatype'] = 16
    if varr.vdtype == 'float64': 
        header['datatype'] = 64

    # units should be mm and sec
    header['xyzt_units'] = 2
    # actually updating header
    varr.nifti_header = header    

    return varr

def _get_pixdims(pd):
    """Return nifti pixdims in [read, phase, slice] dimensions"""

    mul = 10  # procpar lengths are in cm, we need mm
    if pd['apptype'] in ['im2D','im2Dfse','im2Dcs','im2Dfsecs']: 
        d0 = float(pd['lro'])/(int(pd['np'])//2)*mul
        d1 = float(pd['lpe'])/int(pd['nv'])*mul
        d2 = float(pd['thk'])+float(pd['gap'])
        d = (d0,d1,d2)
    elif pd['apptype'] in ['im2Depi','im2Depics']: 
        d0 = float(pd['lro'])/(int(pd['nread'])//2)*mul
        d1 = float(pd['lpe'])/int(pd['nphase'])*mul
        d2 = float(pd['thk'])+float(pd['gap'])
        d = (d0,d1,d2)
    elif pd['apptype'] in ['im3D','im3Dcs','im3Dfse']: 
        d0 = float(pd['lro'])/(int(pd['np'])//2)*mul
        d1 = float(pd['lpe'])/int(pd['nv'])*mul
        d2 = float(pd['lpe2'])/int(pd['nv2'])*mul
        d = (d0,d1,d2)
    else:
        raise(Exception('apptype not implemented in _get_pixdim'))

    return d

def _get_translations(pd):
    """Return translations along axes (read, phase,slice)"""
    # try it naively
    #TODO
    return (0,0,0)

def _qform_affine(varr):
    """Return qform affine if dat is in local space"""
    if varr.space == 'local':
        pass
    else:
        warnings.warn('Not in local space, affine probably incorrect')
    pixdim = _get_pixdims(varr.pd)
    trans = _get_translations(varr.pd)
    rot = _qform_rot_matrix(varr.pd)
    trans = _translation_to_xyz(trans, rot)
    pixdim_matrix = np.eye(3)*pixdim
    affine = np.zeros((4,4))
    affine[0:3,0:3] = rot @ pixdim_matrix 
    affine[0:3,3] = trans
    affine[3,3] = 1
    return affine


def _qform_rot_matrix(pd):
    """Return rotation matrix for qform affine"""
    # try without
    #TODO
    psi = float(pd['psi']) * 2*np.pi / 360
    phi = float(pd['phi']) * 2*np.pi / 360
    t = float(pd['theta']) * 2*np.pi / 360
    
    #Rotations from Euler angles
    rot_x = np.array([[1,       0,       0],\
                    [0, cos(psi),-sin(psi)],\
                    [0, sin(psi), cos(psi)]])

    rot_y = np.array([[cos(phi),0,sin(phi)],\
                    [0,         1,0      ],\
                    [-sin(phi),0, cos(phi)]])

    rot_z = np.array([[cos(t),-sin(t),0],\
                    [sin(t), cos(t), 0],\
                    [0,     0,      1]])
    m = rot_x @ rot_y @ rot_z 
    print(m)

    #matrix = np.eye(3)
    return m

def _translation_to_xyz(t, rot_matrix):
    """Return translation vector in x,y,z from ro,pe,slc"""
    #TODO
    return t
