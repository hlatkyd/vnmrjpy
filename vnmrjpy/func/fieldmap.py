import vnmrjpy as vj
import numpy as np
import lmfit
from vnmrjpy.core.utils import vprint
import itertools
"""
Generate fieldmap from a set of gradient echo images
"""

def _make_grid_x3d(shape3d, gridpoints):
    """Return 4d numpy array: a vector at each point in a 3d volume"""
    newshape = list(shape3d) + [gridpoints.shape[0]]
    arr = np.zeros(newshape)
    arr[...] = gridpoints
    return arr

def _make_phase_set(phasedata, method='triple_echo'):
    """Return all possible phase wrapping combinations

    Args:
        phasedata -- numpy ndarray of dim (x,y,z, time, rcvrs) with phase data
    Return:
        phasedata_list -- list of possible phase data in all possible
                                cases of phase wrapping

    Note: This phase ordering rule is also used in _get_best_fit function
    """
    # only implemented for 3 TE points
    if phasedata.shape[3] != 3:
        raise(Exception('Only the triple echo case is implemented'))
    if method == 'triple_echo': 
        p0 = phasedata[...,0,:]
        p1 = phasedata[...,1,:]
        p2 = phasedata[...,2,:]
        #See ref [1] in make_fieldmap()
        #case 1
        case1 = phasedata
        phasedata_list = [case1]
        #case 2
        case2 = np.stack([p0,p1+2*np.pi,p2+2*np.pi],axis=3)
        phasedata_list.append(case2)
        #case 3
        case3 = np.stack([p0,p1-2*np.pi,p2-2*np.pi],axis=3)
        phasedata_list.append(case3)
        #case 4
        case4 = np.stack([p0,p1,p2+2*np.pi],axis=3)
        phasedata_list.append(case4)
        #case 5
        case5 = np.stack([p0,p1,p2-2*np.pi],axis=3)
        phasedata_list.append(case5)
        #case 6
        case6 = np.stack([p0,p1+2*np.pi,p2],axis=3)
        phasedata_list.append(case6)
        #case 7
        case7 = np.stack([p0,p1-2*np.pi,p2],axis=3)
        phasedata_list.append(case7)
        
        return phasedata_list
    else:
        raise Exception 

def _calc_freq_shift(phase_set, te, method='triple_echo'):
    """Calculate frequency shift at each point for each phase wrapping scenario

    Do linear regression of the form Phase(TE) = c + d * TE

    Args:
        phase_set -- list of phase sets in different phase wrapping cases
        te -- echo times in ms
    Return:
        d_set
        chisqr_set
    """
    if method == 'triple_echo':
        d_set = []
        chisqr_set = []
        shape = phase_set[0].shape
        
        te = np.tile(np.array(te),shape+tuple([1]))
        te = te.swapaxes(3,5)[...,0]
        te_sum = np.sum(te,axis=3,keepdims=True)
        te_sqr_sum = np.sum(te,axis=3,keepdims=True)**2 
        te_sum_sqr = np.sum(te**2, axis=3,keepdims=True)

        for phase in phase_set:
            
            # calculate slope
            d = (np.sum(te * phase, axis=3,keepdims=True) - te_sum *
                        np.sum(phase,axis=3,keepdims=True) ) / \
                    (te_sqr_sum - te_sum_sqr)
            # calculate intercept
            c = np.mean(phase,axis=3,keepdims=True) - \
                        d * np.mean(te,axis=3,keepdims=True)
            # calculate chi-squared
            chisqr = np.sum((phase - d * te -c)**2,axis=3,keepdims=True)
            d_set.append(d)
            chisqr_set.append(chisqr)

        return d_set, chisqr_set
    else:
        raise Exception

def _get_original_phase(d,ind,method='triple_echo'):
    """Return original phase"""
    pass
def _get_best_fit(d, chisqr, method='triple_echo'):
    """Find element in d with lowest chisqr at each voxel 

    Args:
        d
        chisqr
    Return:
        fieldmap
    """
    
    shape = chisqr[0].shape
    fieldmap = np.zeros(shape,dtype='float32')
    loop_dims = [shape[i] for i in [0,1,2,4]]
    for x, y, z, rcvr in itertools.product(*map(range,loop_dims)):

        min_chisqr = min(chisq[x,y,z,0,rcvr])
        fieldmap[x,y,z,0,rcvr] = min_d

    return fieldmap

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
        print('phasedata shape {}'.format(phasedata.shape))
        phase_set = _make_phase_set(phasedata)
        print('phaseset shape {}'.format(phase_set[0].shape))

        d_set, chisqr_set = _calc_freq_shift(phase_set, te)
        fieldmap = _get_best_fit(d_set, chisqr_set)



        # TODO deprecated
        # would be nice if performance was OK
        # consider namedtuples when saving result object into array
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





