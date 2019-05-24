import vnmrjpy as vj
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
import lmfit
from vnmrjpy.core.utils import vprint
import itertools
import matplotlib.pyplot as plt
import copy

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

    Note: This phase ordering rule is also used in _get_fieldmap function
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
        # getting into sec
        #te = np.divide(te,1000)
        te = np.tile(np.array(te),shape+tuple([1]))
        te = te.swapaxes(3,5)[...,0]
        print('te  check {}'.format(te[0,0,0,:,0]))
        te_sum = np.sum(te,axis=3,keepdims=True)
        te_sqr_sum = (np.sum(te,axis=3,keepdims=True))**2 
        te_sum_sqr = np.sum(te**2, axis=3,keepdims=True)

        te_mean = np.mean(te, axis=3, keepdims=True)

        for num, phase in enumerate(phase_set):
            
            # calculate slope
            
            """
            d = (np.sum(te * phase, axis=3,keepdims=True) - te_sum *
                        np.sum(phase,axis=3,keepdims=True) ) / \
                    (te_sum_sqr - te_sqr_sum)

            
            """
            phase_mean = np.mean(phase, axis=3,keepdims=True) 
            d = np.sum((phase-phase_mean)*(te-te_mean), axis=3,keepdims=True) / \
                np.sum((te-te_mean)**2, axis=3, keepdims=True)

            

            # calculate intercept
            c = np.mean(phase,axis=3,keepdims=True) - \
                        d * np.mean(te,axis=3,keepdims=True)
            #c = phase_mean - d * te_mean
            # calculate chi-squared
            print('phase shape here {}'.format(phase.shape))
            chisqr = np.sum((phase - d * te -c)**2,axis=3,keepdims=True)
            
            #chisqr = np.sum((phase-(d*te+c)**2/(d*te+c)),axis=3,keepdims=True)
            #chisqr = np.sum((phase-(d*te+c)**2),axis=3,keepdims=True)

            # hack here: set chisqr high where d is negative
            chisqr[d < 0] = 10

            # append to list
            d_set.append(d)
            chisqr_set.append(chisqr)
            
            # TODO fit check, del this
            plt.subplot(1,7,num+1)
            plt.plot(c[45,10,45,:,1] + d[45,10,45,:,1] * te[45,10,45,:,1],color='b')
            plt.plot(d[45,10,45,:,1] * te[45,10,45,:,1],color='g')
            plt.plot(phase_set[num][45,10,45,:,1],'ro')
            plt.title(str(num))
            plt.ylim((-2*np.pi, 2*np.pi))
        plt.show()

        return d_set, chisqr_set
    else:
        raise Exception

def _get_indice_map(chisqr_set, method='triple_echo'):
    """Find element with lowest chisqr at each voxel 

    Args:
        chisqr_set
    Return:
        indice_arr
    """
    
    if method == 'triple_echo':
        
        #make chisqr array of dims [x,y,z,0,rcvr,chisqr]
        chisqr_arr = np.stack(chisqr_set,axis=5)
        indice_arr = np.argmin(chisqr_arr,axis=5)
        return indice_arr
    else:
        raise Exception

def _get_fieldmap(d_set, indice_arr, method='triple_echo'):
    """Find best linear fit and return final field map in angular frequency units"""
    
    if method == 'triple_echo':
        d_arr = np.stack(d_set,axis=0)
        #d_arr = np.moveaxis(d_arr, [5,0,1,2,3,4],[0,1,2,3,4,5])
        fieldmap = np.choose(indice_arr, d_arr)
        
        return fieldmap
    else:
        raise Exception

def _median_filter(fieldmap):

    print('fialdmap shape {}'.format(fieldmap.shape))
    for rcvr in range(fieldmap.shape[4]):
        arr = copy.copy(fieldmap[...,0,rcvr])
        fieldmap[...,0,rcvr] = median_filter(arr,size=(3,1,3))

        plt.subplot(1,2,1)
        plt.imshow(arr[:,10,:])
        plt.subplot(1,2,2)
        plt.imshow(fieldmap[:,10,:,0,rcvr])
        plt.show()

    return fieldmap

#TODO make filter size based on real voxel size maybe?
def _gaussian_filter(fieldmap, kernel=2):
    
    pass

def _combine_receivers(fieldmap_rcvr, magnitude_rcvr):

    fieldmap = np.sum(fieldmap_rcvr * magnitude_rcvr, axis=4) / \
                np.sum(magnitude_rcvr,axis=4)
    return fieldmap
def make_fieldmap(varr, mask=None, selfmask=True, combine_receivers=True, \
                    method='triple_echo'):
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
    
    if mask == None and selfmask==True:
        varr, mask = vj.func.mask(varr, mask_out=True)

    if method=='triple_echo':
        # checking for echos
        time_dim = varr.data.shape[3]
        # calcin milliseconds
        te = [float(i)*1000 for i in varr.pd['te']]
        print('timedim {}; te {}'.format(time_dim, te))
        phasedata = np.arctan2(np.imag(varr.data),np.real(varr.data))
        magnitudedata = np.abs(varr.data)
        phasedata.astype('float32')
        print('phasedata shape {}'.format(phasedata.shape))
        phase_set = _make_phase_set(phasedata, method=method)
        print('phaseset shape {}'.format(phase_set[0].shape))

        d_set, chisqr_set = _calc_freq_shift(phase_set, te, method=method)
        indice_arr = _get_indice_map(chisqr_set, method=method)
        fieldmap = _get_fieldmap(d_set, indice_arr, method=method)
        print(fieldmap.shape)
        print(mask.shape)
        if type(mask) != type(None):
            fieldmap = fieldmap * mask
        print(fieldmap.shape)

        #fieldmap = _combine_receivers(fieldmap,magnitudedata)

        #fieldmap = _median_filter(fieldmap)

        #fieldmap = _gaussian_filter(fieldmap, kernel=2)

        print('indice arr shape {}'.format(indice_arr.shape))

        plt.subplot(3,4,1)
        plt.imshow(fieldmap[:,10,:,0,1],cmap='gray',vmin=0,vmax=4)
        #plt.imshow(fieldmap[:,10,:,0],cmap='gray',vmin=0,vmax=4)
        plt.subplot(3,4,2)
        plt.imshow(fieldmap[:,10,:,0,2],cmap='gray',vmin=0,vmax=4)
        plt.subplot(3,4,3)
        plt.imshow(indice_arr[:,10,:,0,1],cmap='gray')
        plt.subplot(3,4,4)
        plt.imshow(d_set[0][:,10,:,0,1],cmap='gray')
        plt.subplot(3,4,5)
        plt.imshow(phasedata[:,10,:,0,1],cmap='gray')
        plt.subplot(3,4,6)
        plt.imshow(phasedata[:,10,:,1,1],cmap='gray')
        plt.subplot(3,4,7)
        plt.imshow(phasedata[:,10,:,2,1],cmap='gray')
        plt.subplot(3,4,8)
        plt.plot(phasedata[45,10,45,:,1])
        plt.subplot(3,4,9)
        plt.plot(phase_set[3][45,10,45,:,1])
        print(phase_set[3][45,10,45,:,1])
        
        chilist = [ float(i[45,10,45,:,1]) for i in chisqr_set]
        dlist = [ float(i[45,10,45,:,1]) for i in d_set]
        print('chi list')
        print(chilist)
        print('d list')
        print(dlist)
        print('index in arr {}'.format(indice_arr[45,10,45,0,1]))
        #print(fieldmap[45,10,45,0,1])
        print(d_set[3][45,10,45,0,1])
        plt.show()



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





