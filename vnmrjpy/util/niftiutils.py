import vnmrjpy as vj
import nibabel as nib
import numpy as np

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
