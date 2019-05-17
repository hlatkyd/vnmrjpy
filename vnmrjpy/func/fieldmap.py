import vnmrjpy as vj
import numpy as np
import lmfit
from vnmrjpy.core.utils import vprint
"""
Generate fieldmap from a set of gradient echo images
"""

def _make_grid_x3d(shape3d, gridpoints):
    """Return 4d numpy array: a vector at each point in a 3d volume"""
    newshape = list(shape3d) + [gridpoints.shape[0]]
    arr = np.zeros(newshape)
    arr[...] = gridpoints
    return arr

def _make_phase_set(phasedata):
    """Return all possible phase wrapping combinations

    Args:
        phasedata -- numpy ndarray of dim (x,y,z, time, rcvrs) with phase data
    Return:
        phasedata_list -- list of possible phase data in all possible
                                cases of phase wrapping
    """
    # only implemented for 3 TE points
    if phasedata.shape[3] != 3:
        raise(Exception('Only the triple echo case is implemented'))
    
    p0 = phasedata[...,0,:]
    p1 = phasedata[...,1,:]
    p2 = phasedata[...,2,:]
    #See ref [1] in make_fieldmap()
    #case 1
    case1 = phasedata
    phasedata_list = [case1]
    #case 2
    case2 = np.concatenate([p0,p1+2*np.pi,p2+2*np.pi])
    phasedata_list.append(case2)
    #case 3
    case3 = np.concatenate([p0,p1-2*np.pi,p2-2*np.pi])
    phasedata_list.append(case3)
    #case 4
    case4 = np.concatenate([p0,p1,p2+2*np.pi])
    phasedata_list.append(case4)
    #case 5
    case5 = np.concatenate([p0,p1,p2-2*np.pi])
    phasedata_list.append(case5)
    #case 6
    case6 = np.concatenate([p0,p1+2*np.pi,p2])
    phasedata_list.append(case6)
    #case 7
    case7 = np.concatenate([p0,p1-2*np.pi,p2])
    phasedata_list.append(case7)
    
    return phasedata_list
    
def _calc_phase_slope(phase_set, te):
    """Calculate linear regression line slope at each point in 3d phase map

    """
    d_set = []
    te_arr = np.zeros_like(phase_set)
    te = np.array(te)    
    te_sum = np.sum(te)
    te_sqr_sum = np.sum(te)**2 
    te_sum_sqr = np.sum(te**2)

    for phase in phase_set:
        d = phase[]
        pass

    pass
    
def make_fieldmap(varr,method='triple_echo'):
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

    def _linear_func(x,a,b):
        return a * x + b
    
    if method=='triple_echo':
        # checking for echos
        time_dim = varr.data.shape[3]
        # calcin milliseconds
        te = [float(i)*1000 for i in varr.pd['te']]
        print('timedim {}; te {}'.format(time_dim, te))
        phasedata = np.arctan2(np.imag(varr.data),np.real(varr.data))
        phasedata.astype('float32')
        phase_set = _make_phase_set(phasedata)

        slope_set = _calc_phase_slope(phase_set, te)
        print(phase_set[0].shape)



        """
        # create the model and parameter space
        model = lmfit.Model(_linear_func)
        keys = model.param_names
        vals = [0.0,1.0]
        params_dict = dict(zip(keys, vals))
        shape3d = phasedata.shape[0:3]
        params3d = vj.fit.make_params3d(model, shape3d, **params_dict)
        print('params3d shape {}'.format(params3d.shape))
        print('params3d element {}'.format(params3d[0,0,0]))
        timegrid = np.array([1000*float(i) for i in varr.pd['te']])
        x3d = _make_grid_x3d(shape3d,timegrid)  # time grid is the same everywhere
    
        # for each independent receiver channel
        for rcvr in range(phasedata.shape[4]):

            print('Processing receiver {}'.format(rcvr))
            vprint('Processing receiver {}'.format(rcvr))
            # fit data to each element of the phase set
            for k, phase in enumerate(phase_set):
                print('processing phase {}'.format(k))
                
                y3d = phase[...,rcvr]
            
        
        #return b0map
        """

    else:
        raise(Exception('Not implemented fieldmap generating method'))





