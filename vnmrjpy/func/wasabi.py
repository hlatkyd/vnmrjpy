import vnmrjpy as vj
import numpy as np
from scipy.optimize import least_squares


def WASABI(varr,mask=None):
    """WASABI B0 and B1 mapping function

    Args:
        varr (vj.varray) -- input varray to fit
        mask (vj.varray) -- mask where fitting should be done
    Return:
        b0map (vj.varray) -- 
        b1map (vj.varray) -- 
        c (vj.varray) -- fit parameter
        d (vj.varray) -- fit parameter

    Use with specifically designed sequence. See reference paper for theory.

    Ref.: Schuenke et al.: Simultaneous mapping of water shift and B1
        (WASABI) - Application to field-inhomogenty correction of CESTMRI
        data.
    """
    # -----------------------helper fuctions-----------------------------------
    def _z_spectrum(B1,dw,c,d):
        """Return curve of Z spectrum as in ref paper"""
        #TODO check this
        tp = varr.pd['']  # pulse duration
    
        delta_w = w_rf - w_larmor
        z = np.absolute(c - d * np.sin(np.arctan((gamma*B1)/delta_w-dw))**2 *\
                np.sin(np.sqrt((gamma*B1)**2+(delta_w-dw)**2) * tp/2)**2)
        return z
    
    
    # checking whether data is actually used for wasabi
    if len(varr.pd['mtfrq']) == 1:
        raise(Exception('Only one MT frequence is found. Cannot use WASABI'))
    # checking for imagespace
    if varr.vdtype != 'imagespace':
        print('Waring: Input varray is not in imagespace.'
                'Trying to_imagespace() and proceeding...')
        varr.to_imagespace()
   
    #-------------------------- actual satart----------------------------------
    f_larmor = float(varr.pd['reffrq']) * 10**6
    w_larmor = 2 * np.pi * f_larmor
    #f_larmor = 2*np.pi*42.578*9.4  # Mhz
    flip_mt = float(varr.pd['flipmt'])  # MT flip angle
    #TODO masking for noise
    
    # normalize intenisties

    #lookup-table

 
    return 0
