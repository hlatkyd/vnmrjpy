import vnmrjpy as vj
import numpy as np
import nibabel as nib
import warnings
from math import sin, cos

def _set_nifti_header(varr):
    """Set varray.nifti_header attribute for the current space"""
    # init empty header
    header = nib.nifti1.Nifti1Header()
    # use qform only

    # Transform, so q_affine is diagonal
    if varr.space == 'anatomical':

        qfac = 1
        dim_count = len(varr.data.shape)
        header['dim'][0] = dim_count
        for i in range(dim_count):
            header['dim'][i+1] = varr.data.shape[i]
        # setting pixdim based on procpar data
        d = _get_pixdims(varr.pd)
        swap,flip = vj.core.transform._anatomical_swaps(varr.pd)
        d_swap = [i[1] for i in sorted(zip(swap,d))]
        # rotate pixdim
        # see qfac in doc
        header['pixdim'][0] = qfac
        for i in range(3):
            header['pixdim'][i+1] = d_swap[i]

        # set local dim info
        header.set_dim_info(freq=swap[1],phase=swap[0],slice=swap[2])

        # setting qform
        header['qform_code'] = 1
        q_affine = _qform_affine(varr,qfac)
        
        header.set_qform(q_affine,code=1)
   
    # Original without any transformation 
    elif varr.space == 'local':
        
        qfac = 1  # qfac: -1 if left handed
        dim_count = len(varr.data.shape)
        header['dim'][0] = dim_count
        for i in range(dim_count):
            header['dim'][i+1] = varr.data.shape[i]
        # setting pixdom based on procpar data
        d = _get_pixdims(varr.pd)
        
        # see qfac in doc
        header['pixdim'][0] = qfac
        for i in range(3):
            header['pixdim'][i+1] = d[i]

        # set local dim info
        header.set_dim_info(freq=1,phase=0,slice=2)

        # setting qform
        header['qform_code'] = 1
        q_affine = _qform_affine(varr,qfac)
        
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

def _qform_affine(varr,qfac):
    """Return qform affine"""
    if varr.space == 'local':
        pixdim = _get_pixdims(varr.pd)
        trans = _get_translations(varr.pd)
        rot = _qform_rot_matrix(varr.pd)
        trans = _translation_to_xyz(trans, rot)
        pixdim_matrix = np.eye(3)*pixdim
        pixdim_matrix[2,2] = pixdim_matrix[2,2]*qfac

        affine = np.zeros((4,4))
        affine[0:3,0:3] = rot @ pixdim_matrix 
        affine[0:3,3] = trans
        affine[3,3] = 1
    elif varr.space == 'anatomical':
        # make the affine diagonal in this case
        warnings.warn('Not in local space, affine probably incorrect')
        pixdim = _get_pixdims(varr.pd)
        trans = _get_translations(varr.pd)
        rot = _qform_rot_matrix(varr.pd)
        pixdim = _pixdim_to_xyz(pixdim, varr.pd)
        # TODO
        #trans = _translation_to_xyz(trans, rot)
        # just make translations 0 for now
        trans = [0,0,0]

        pixdim_matrix = np.eye(3)*pixdim
        pixdim_matrix[2,2] = pixdim_matrix[2,2]*qfac
        aff_3x3 = np.eye(3) * pixdim_matrix

        affine = np.zeros((4,4))
        affine[0:3,0:3] = aff_3x3
        affine[0:3,3] = trans
        affine[3,3] = 1
        #print(affine)
    else:
        raise(Exception('Cant calc q_form affine in this space'))
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

    #matrix = np.eye(3)
    return m

def _translation_to_xyz(t, rot_matrix):
    """Return translation vector in x,y,z from ro,pe,slc"""
    #TODO
    return t

def _pixdim_to_xyz(pixdim, pd):

    swap, flip = vj.core.transform._anatomical_swaps(pd)
    pixdim = [i[1] for i in sorted(zip(swap,pixdim))]
    return pixdim

