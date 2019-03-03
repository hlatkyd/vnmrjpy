import vnmrjpy as vj
import numpy as np

def calc_array_length(fid_data_shape,procpar):
    """Find the length of acquisition parameter array"""

    p = vj.io.ProcparReader(procpar).read()
    rcvrs = p['rcvrs'].count('y')
    blocks = fid_data_shape[0]
    seqcon = p['seqcon']
    # seqcon chars: echo,slice,pe1,pe2,pe3
    s_pos = [pos for pos, i in enumerate(seqcon) if i=='s']
    s_ind = np.array([0,0,0,0,0])
    # find dimensions according to seqcon
    pe = int(p['nv'])
    try:
        ne = int(p['ne'])
    except:
        ne = 1
    try:
        slc = int(p['ns'])
    except:
        slc = 1
    try:
        pe2 = int(p['nv2'])
    except:
        pe2 = 1
    # pe3 is only supported in csi
    pe3 = 1
    
    seqcon_dims = [ne,slc,pe,pe2,pe3]
    s_ind[s_pos] = 1
    blocks_wo_array = 0
    for i in range(5):
        blocks_wo_array+=seqcon_dims[i]*s_ind[i]
    if blocks_wo_array != 0:
        l = blocks // blocks_wo_array - rcvrs
    else:
        l = 1
    if l > 0:
        return l
    else:
        return 1
