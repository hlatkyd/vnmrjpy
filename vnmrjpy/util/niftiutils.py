import vnmrjpy as vj
import nibabel as nib
import numpy as np

def make_6d(data5d, output_type='magn-phase',input_rcvrdim=5):
    """Return 6D data from complex 5D

    Input should be the usual output of KspaceMaker of ImageSpaceMaker,
    with the forat [receivers, x, y, z, time].
    Return data in [x,y,z,t,receiver,magnitude/phase] format or
    real-imaginynary

    Args:
        data5d (np.ndarray) -- input 5d data
        output_type (string) -- either 'magn-phase' or 'real-imag'
    Return:
        data6d
    """
    olddims = [0,1,2,3,4]
    newdims = [1,2,3,4,0]
    if input_rcvrdim == 0:
        data5d = np.moveaxis(data5d,newdims,olddims)
    if output_type == 'magn-phase':
        magn = np.absolute(data5d)
        phase = np.arctan2(np.imag(data5d),np.real(data5d))
        data6d = np.concatenate([magn[...,np.newaxis],phase[...,np.newaxis]],axis=-1)
    elif output_type == 'real-imag':
        real = np.real(data5d)
        imag = np.imag(data5d)
        data6d = np.concatenate([real[...,np.newaxis],imag[...,np.newaxis]],axis=-1)
    else:
        raise(Exception('Wrong "output_type" spec'))
    return data6d

def make_local_matrix(p):
    """Make matrix for nifit header based on procpar dict p

    local coords are [pahse, read, slice/phase2, time]
    """
    # the 10 multiplier is from cm -> mm conversion
    mul = 10
    if p['apptype'] in ['im2Depi']:
        dpe = float(p['lpe']) / (int(p['nphase'])) * mul
        dro = float(p['lro']) / (int(p['nread'])//2) * mul
        ds = float(p['thk'])+float(p['gap'])
        matrix = [int(p['phase']),int(p['nread'])//2,int(p['ns'])]
        dim = [dpe,dro,ds]
    elif p['apptype'] in ['im2Dfse', 'im2D', 'im2Dcs']:
        dpe = float(p['lpe']) / (int(p['nv'])) * mul
        dro = float(p['lro']) / (int(p['np'])//2) * mul
        ds = float(p['thk'])+float(p['gap'])
        matrix = [int(p['nv']),int(p['np'])//2,int(p['ns'])]
        dim = [dpe,dro,ds]
    elif p['apptype'] in ['im3D','im3Dshim', 'im3Dcs']:
        dpe = float(p['lpe']) / (int(p['nv'])) * mul
        dro = float(p['lro']) / (int(p['np'])//2) * mul
        dpe2 = float(p['lpe2']) / (int(p['nv2'])) * mul
        matrix = [int(p['nv']),int(p['np'])//2,int(p['nv2'])]
        dim = [dpe,dro,dpe2]
    
    # return dim as well?
    return matrix, dim

def swapdim(swaparr, dim):
    """Swap dimensions array to fit data created with swaparray"""

    a = sorted(zip(swaparr,dim))
    return [a[i][1] for i in range(3)]
