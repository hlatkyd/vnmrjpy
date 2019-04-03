import vnmrjpy as vj
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import itertools
import copy
import matplotlib.pyplot as plt
import sys
from lmfit import minimize, Parameters

def z_spectrum(delta_w, b1, dw, c, d, tp=0.005, f_scale=400, gamma=42.57747):
    """Return curve of Z spectrum for WASABI

    Args:
        delta_w -- magnetisation transfer offset frequencies
        b1 -- B1 transverse field in microT
        dw -- freq offset made by B0 offset
        c -- fit parameter
        d -- fit parameter
        
        tp -- offset pulse duration
        gamma -- giromagnetic ratio for H1
        f_scale -- frequency scaling factor (only for easier parameter fitting)
    """
    z = np.absolute(c - d*np.sin(np.arctan((gamma*b1)/delta_w-dw*f_scale))**2\
         *np.sin(np.sqrt((gamma*b1)**2+(delta_w-dw*f_scale)**2) * tp/2)**2)
    return z

def WASABI(varr, norm, mask='none'):
    """WASABI B0 and B1 mapping function

    Args:
        varr (vj.varray) -- input varray to fit
        norm (vj.varray) -- additional data for normalization
        mask (np.ndarray) -- data mask where fitting should be done
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
    
    def _residual(params, x, data, eps_data):
        """Residual for least-squares fitting"""
        b1 = params['b1']    
        dw = params['dw']    
        c = params['c']    
        d = params['d'] 
        model = z_spectrum(x,b1,dw,c,d,tp=tp,f_scale=f_scale)
        return (data-model) / eps_data   


    # checking whether data is actually used for wasabi
    if len(varr.pd['mtfrq']) == 1:
        raise(Exception('Only one MT frequence is found. Cannot use WASABI'))
    # checking for imagespace
    if varr.vdtype != 'imagespace':
        print('Waring: Input varray is not in imagespace.'
                'Trying to_imagespace() and proceeding...')
        varr.to_imagespace()
   
    #-------------------------- actual satart----------------------------------

    gamma = 42.57747  # giromagnetic ratio in Hz/T
    tp = float(varr.pd['pmt'])/10**6  # is in microsec initially, set to sec
    flip1 = float(varr.pd['flip1'])*2*np.pi/360  #flip angle set to rad
    flipmt = float(varr.pd['flipmt'])*2*np.pi/360  #mt flip angle set to rad
    B1 = flipmt / (2*np.pi*gamma*tp) # B1 in microT
    eps_data = 1 #TODO estimated uncertainty of data. get from noise
    delta_w = []  # freq offsets
    for item in varr.pd['mtfrq']:
        delta_w.append(float(item))
    f_scale = 400  # frequency scaling, jsut for parameter estimation help

    print('flip1 {}'.format(flip1))
    print('tp : {}'.format(tp))
    print('B1 : {}'.format(B1))

    # make magnitude image
    magn_orig = vj.core.recon.ssos(varr.data) 
    #magn = copy.deepcopy(magn)
    magn_norm = vj.core.recon.ssos(norm.data)
    #mask if available
    if mask != 'none':
        magn_orig = magn_orig*mask
    # normalize data
    magn = np.divide(magn_orig,magn_norm) 
    # init output
    B0map = np.zeros_like(magn)
    B1map = np.zeros_like(magn)
    c_map = np.zeros_like(magn)
    d_map = np.zeros_like(magn)
    # std error maps
    B0map_err = np.zeros_like(magn)
    B1map_err = np.zeros_like(magn)
    c_map_err = np.zeros_like(magn)
    d_map_err = np.zeros_like(magn)
    # setting up lmfit
    params = Parameters()
    params.add('b1',value=3.8,min=0.5,max=7)
    params.add('dw',value=0,min=-2,max=2)
    params.add('c',value=0.5,min=0.1,max=1)
    params.add('d',value=1,min=0.3,max=3)
    # iterate over every voxel in space
    x_dim, y_dim, z_dim = varr.data.shape[0:3]
    for x, y, z in itertools.product(*map(range, (x_dim,y_dim,z_dim))):
        if mask[x,y,z,0] == 0:
            continue
        else:
            xdata = delta_w
            ydata = magn[x,y,z,:]
            try:
                out = minimize(_residual, params, args=(xdata,ydata,eps_data))
                # output vals
                B1map[x,y,z,0] = out.params['b1'].value
                B0map[x,y,z,0] = out.params['dw'].value
                c_map[x,y,z,0] = out.params['c'].value
                d_map[x,y,z,0] = out.params['d'].value
                # output errors
                B1map_err[x,y,z,0] = out.params['b1'].stderr
                B0map_err[x,y,z,0] = out.params['dw'].stderr
                c_map_err[x,y,z,0] = out.params['c'].stderr
                d_map_err[x,y,z,0] = out.params['d'].stderr
            except Exception:
                continue
            except KeyboardInterrupt:
                sys.exit(0)
    # creating varray
    varr_b0 = copy.copy(norm)
    varr_b1 = copy.copy(norm)
    varr_c = copy.copy(norm)
    varr_d = copy.copy(norm)
    # create error varrs
    varr_b0_err = copy.copy(norm)
    varr_b1_err = copy.copy(norm)
    varr_c_err = copy.copy(norm)
    varr_d_err = copy.copy(norm)

    varr_b0.data = np.array(B0map,dtype='float32')
    varr_b0.intent = 'B0_map'
    varr_b1.data = np.array(B1map,dtype='float32')
    varr_b1.intent = 'B1_map'
    varr_c.data = np.array(c_map,dtype='float32')
    varr_c.intent = 'wasabi_c'
    varr_d.data = np.array(d_map,dtype='float32')
    varr_d.intent = 'wasabi_d'
    # errors
    varr_b0_err.data = np.array(B0map_err,dtype='float32')
    varr_b1_err.data = np.array(B1map_err,dtype='float32')
    varr_c_err.data = np.array(c_map_err,dtype='float32')
    varr_d_err.data = np.array(d_map_err,dtype='float32')
    return ((varr_b0, varr_b1, varr_c, varr_d) , \
                (varr_b0_err, varr_b1_err, varr_c_err, varr_d_err))
