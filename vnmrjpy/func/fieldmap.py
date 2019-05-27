import vnmrjpy as vj
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
from vnmrjpy.core.utils import vprint
import copy

# for hiding zero-divide warnigns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Generate fieldmap from a set of gradient echo images
"""

def make_fieldmap(varr, mask=None, selfmask=True, combine_receivers=True, \
                    method='triple_echo'):
    """Generate B0 map from gradient echo images.

    Args:
        varr -- input gradient echo image set with different
                echo time acquisitions at time dimension
        method -- triple_echo: see ref [1]
        mask -- ndarray of [0 or 1] with same shape of varr.data in spatial dims
        selfmask -- Boolean, set True to create mask based on magnitude data
    Return:
        fieldmap -- vj.varray with data attribute updated to the B0 map

    Refs:
    [1] Windischberger et al.: Robust Field Map Generation Using a Triple-
    Echo Acquisition, JMRI, (2004)
    """

    if mask == None and selfmask==True:
        varr, mask = vj.func.mask(varr, mask_out=True)

    if method=='triple_echo':

        kernel_size = _get_gaussian_kernel(varr)
        # checking for echos
        time_dim = varr.data.shape[3]
        # calcin milliseconds
        te = [float(i)*1000 for i in varr.pd['te']]
        phasedata = np.arctan2(np.imag(varr.data),np.real(varr.data))
        magnitudedata = np.abs(varr.data)
        phasedata.astype('float32')
        phase_set = _make_phase_set(phasedata)

        d_set, c_set, residual_set = _calc_freq_shift(phase_set,te)
        indice_arr = _get_indice_map(residual_set)
        c_map = _get_const(c_set, indice_arr)
        # pre field map without filters and receivers not combined
        fieldmap = _get_fieldmap(d_set, indice_arr)

        #TODO combine receivers somehow. magnitue weighting is not too ok
        #fieldmap = _combine_receivers(fieldmap,magnitudedata)
        #c_map = _combine_receivers(c_map, magnitudedata)

        fieldmap = _median_filter(fieldmap, mask=mask)

        fieldmap = _gaussian_filter(fieldmap, kernel=kernel_size, mask=mask)

        # creating varray
        varr.data = fieldmap
        return varr

    else:
        raise(Exception('Not implemented fieldmap generating method'))

def _get_gaussian_kernel(varr):

    if type(varr.nifti_header) == type(None):
        varr = varr.set_nifti_header()
    affine = varr.nifti_header.get_qform()
    kernel = [1/i for i in [affine[0,0],affine[1,1],affine[2,2]]]
    return kernel

#TODO is this used anymore??
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

    Note: This phase ordering rule is also used in _get_fieldmap function
    """
    # only implemented for 3 TE points
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

def _calc_freq_shift(phase_set, te):
    """Calculate frequency shift at each point for each phase wrapping scenario

    Do linear regression of the form Phase(TE) = c + d * TE

    Args:
        phase_set -- list of phase sets in different phase wrapping cases
        te -- echo times in ms
    Return:
        d_set
        c_set
        residual_set
    """
    d_set = []
    c_set = []
    residual_set = []
    shape = phase_set[0].shape
    te = np.array(te,dtype=float)
    for num, phase in enumerate(phase_set):
        
        (x,y,z,t,rcvr) = phase.shape
        # reshape to accomodate polyfit vectorization
        phase = np.reshape(np.swapaxes(phase, 0,3),(t,-1))
        out = np.polyfit(te, phase, 1, full=True)            
        d,c = out[0]
        res = out[1]
        # reshape back to otiginal
        d = np.swapaxes(np.reshape(d,(1,y,z,x,rcvr)),0,3)
        c = np.swapaxes(np.reshape(c,(1,y,z,x,rcvr)),0,3)
        res = np.swapaxes(np.reshape(res,(1,y,z,x,rcvr)),0,3)

        # hack: just make the loss large where d is negative
        res[d < 0] = 10000
        # append to list
        d_set.append(d)
        c_set.append(c)
        residual_set.append(res)

    return d_set, c_set, residual_set

def _get_indice_map(chisqr_set):
    """Find element with lowest chisqr at each voxel """
    #make chisqr array of dims [x,y,z,0,rcvr,chisqr]
    chisqr_arr = np.stack(chisqr_set,axis=5)
    indice_arr = np.argmin(chisqr_arr,axis=5)
    return indice_arr

def _get_fieldmap(d_set, indice_arr):
    """Find best linear fit and return final field map in angular frequency units"""
    
    d_arr = np.stack(d_set,axis=0)
    #d_arr = np.moveaxis(d_arr, [5,0,1,2,3,4],[0,1,2,3,4,5])
    fieldmap = np.choose(indice_arr, d_arr)
    
    return fieldmap

def _get_const(c_set, indice_arr):

    c_arr = np.stack(c_set,axis=0)
    c_map = np.choose(indice_arr, c_arr)
    return c_map

def _median_filter(fieldmap, mask=None):

    for rcvr in range(fieldmap.shape[4]):
        arr = copy.copy(fieldmap[...,0,rcvr])
        # TODO slicing dynamic
        fieldmap[...,0,rcvr] = median_filter(arr,size=(3,1,3))

    return fieldmap

#TODO make filter size based on real voxel size maybe?
def _gaussian_filter(fieldmap, kernel=2, mask=None):

    for rcvr in range(fieldmap.shape[4]):
        arr = copy.copy(fieldmap[...,0,rcvr])
        #fieldmap[...,0,rcvr] = median_gaussian(arr,size=(3,1,3))
        if type(mask) == type(None):
            fieldmap[...,0,rcvr] = gaussian_filter(arr,sigma=kernel)
        elif type(mask) != type(None):
            mask_rcvr = mask[...,0,rcvr]    
            arr = gaussian_filter(arr * mask_rcvr,sigma=kernel)
            arr /= gaussian_filter(mask_rcvr,sigma=kernel)
            arr[mask_rcvr == 0] = 0
            fieldmap[...,0,rcvr] = arr

    return fieldmap

#TODO this is low performing
def _combine_receivers(fieldmap_rcvr, magnitude_rcvr):

    return np.sum(fieldmap_rcvr * magnitude_rcvr, axis=4) / \
                np.sum(magnitude_rcvr,axis=4)






