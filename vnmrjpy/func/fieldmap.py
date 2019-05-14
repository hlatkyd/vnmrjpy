import vnmrjpy as vj
import numpy as np

"""
Generate fieldmap from a set of gradient echo images
"""

def fieldmapgen(varr,method='triple_echo'):
    """Generate B0 map from gradient echo images

    Args:
        varr -- input gradient echo image set with different
                echo time acquisitions at time dimension
        method -- triple_echo: see ref [1]
    Return:
        b0 -- vj.varray of b0 fieldmap

    Refs:
    [1] Windischberger et al.: Robust Field Map Generation Using a Triple-
    Echo Acquisition, JMRI, (2004)
    """

    if method=='triple_echo':
        # checking for echos
        time_dim = varr.data.shape[3]
        te = varr.pd['te']
        print('timedim {}; te {}'.format(time_dim, te))

    else:
        raise(Exception('Not implemented fieldmap generating method'))
