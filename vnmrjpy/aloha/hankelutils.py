import numpy as np
import matplotlib.pyplot as plt
import copy
import vnmrjpy as vj
import numba
#import cupy as cp
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
"""
Functions for handling Hankel matrices in various ALOHA implementations

"""
DTYPE = 'complex64'

def avg_xy_fibers(fiber_x,fiber_y,weight_x,weight_y):
    """Weighted average of same fiber completed with different weights.

    In kx-ky completion the weighing is done sequentially in x and y
    directions at a fixed stage

    Args:
        fiber_x,fiber_y (np.ndarray) -- unweighted fibers
        weights_x,weight_y (np.ndarray) -- kspace weights
    Return:
        fiber (np.ndarray) -- averaged fiber
    """
    wx = np.absolute(weight_x)
    wy = np.absolute(weight_y)
    fiber = (wx*fiber_x+wy*fiber_y)/(wx+wy)
    return fiber

def replace_zerofreq_kx(fiber,lines):
    """put back old lines where division by zero occours"""

    ind = fiber.shape[1]//2
    radius = (lines.shape[1]-1)//2
    print('radius {}'.format(radius))
    fiber[:,ind-radius:ind+radius+1,:] = lines
    return fiber

def replace_zerofreq_ky(fiber,lines):

    ind = fiber.shape[2]//2
    radius = (lines.shape[2]-1)//2
    fiber[:,:,ind-radius:ind+radius+1] = lines
    return fiber
    

# this is to be deprecated, but works fine
def construct_hankel_2d(slice3d,rp):
    """Make Hankel matrix from 2d+rcvrs data

    This is the old implementation
    INPUT: slice2d_all_rcvrs : numpy.array[receivers, slice]
    OUTPUT: hankel : numpy.array (m-p)*(n-q) x p*q*rcvrs
    """
    slice3d = np.array(slice3d, dtype='complex64')
    # annihilating filter size
    (p,q) = rp['filter_size']
    # hankel m n
    (m,n) = slice3d.shape[1:3]

    for rcvr in range(slice3d.shape[0]):

        slice2d = slice3d[rcvr,...]

        #make inner hankel list
        for j in range(0,n):
            # iterate in outer hankel elements
            for k in range(0,p):
                # iterate inner hankel columns
                col = np.expand_dims(slice2d[k:m-p+k+1,j],axis=1)
                if k == 0:
                    cols = col
                else:
                    cols = np.concatenate([cols,col],axis=1)
            if j == 0:
                hankel = np.expand_dims(cols,axis=0)
            else:
                hankel = np.concatenate(\
                    [hankel, np.expand_dims(cols,axis=0)], axis=0)

        # make outer hankel
        for i in range(q):
            #col = cp.hstack([hankel[i:n-q+i,:,:]])
            col = np.vstack([hankel[k,:,:] for k in range(i,n-q+i+1)])
            if i == 0:
                cols = col
            else:
                cols = np.concatenate([cols,col], axis=1)
        # concatenating along the receivers
        if rcvr == 0:
            hankel_full = cols
        else:
            hankel_full = np.concatenate([hankel_full, cols], axis=1)

    return hankel_full

def lvl2_hankel_average(hankel_full,filter_shape, fiber_shape):

    (rcvrs,m,n) = fiber_shape
    (p,q) = filter_shape
    inner_shape = (m-p+1,p)
    outer_shape = (n-q+1,q)

    # make convolution kernel
    kernel = np.fliplr(np.eye(outer_shape[0],dtype='complex64'))
    # shape of 1 hnakel without receiver concat
    oldshape = (hankel_full.shape[0],hankel_full.shape[1]//rcvrs)
    newshape = (inner_shape[0],outer_shape[0],inner_shape[1],outer_shape[1])
    for rcvr in range(rcvrs):

        hankel = hankel_full[:,p*q*rcvr:p*q*(rcvr+1)]
        print('hankel shape rcvr {}'.format(hankel.shape))
        hankel = np.reshape(hankel,newshape,order='f')
        hankel = np.swapaxes(hankel,1,2)
        # now axis 0,1 represents the inner blocks, 2,3 is the outer
        for ind in range(n):  # n is also the number of antidiagonals

            # find each i, j pair for ind
            pairs = [[ind-j,j] for j in range(ind+1) if (j < q and ind-j < n-q+1)]

            list_to_avg = [hankel[...,pair[0],pair[1]] for pair in pairs]
            avg = np.zeros_like(hankel[...,0,0],dtype='complex64') 
            for inner_matrix in list_to_avg:
            
                avg += fftconvolve(inner_matrix,kernel,mode='same') 
            # divide by the weights
            avg = avg / len(pairs)

            for pair in pairs:

                hankel[...,pair[0],pair[1]] = avg

        hankel = np.reshape(np.swapaxes(hankel,1,2),oldshape,order='f')
        hankel_full[:,p*q*rcvr:p*q*(rcvr+1)] = hankel

    return hankel_full

def lvl2_hankel_weights(filter_shape,fiber_shape):
    """Make weights for the full 2 level hankel matrix.

    Each element of result gives the multiplicity of an element in the 2-level
    hankel structure

    Args:
        filter_shape (tuple) -- ALOHA annihilating filter shape
        fiber_shape (tuple) -- 3D filter shape (receiver,dim1,dim2)
    Return:
        full_weights
    """
    (rcvrs,m,n) = fiber_shape
    (p,q) = filter_shape
    (x1,y1) = (m-p+1,p)  # inner hankel shape
    (x2,y2) = (n-q+1,q)  # outer block shape, if 1 inner hankel is 1 element
    # base weight of an inner hankel
    weights1 = np.array([[min(y1,i+j+1) for j in range(y1)] for i in range(x1)])
    weights1 = np.minimum(weights1,np.flipud(np.fliplr(weights1)))
    # weights to weight the weights (yeah!)
    weights2 = np.array([[min(y2,i+j+1) for j in range(y2)] for i in range(x2)])
    weights2 = np.minimum(weights2,np.flipud(np.fliplr(weights2)))
    print('weights2 shape {}'.format(weights2.shape))
    full_weights = np.tile(weights1,(x2,y2,1,1))
    # now weight the weights with the weights
    full_weights = full_weights * weights2[...,np.newaxis,np.newaxis]
    full_weights = np.swapaxes(full_weights,1,2)
    full_weights = np.reshape(full_weights,(x1*x2,y1*y2))
    # side-by-side concatenation as per receiverdim
    #full_weights = np.concatenate([full_weights for i in range(rcvrs)],axis=1)
    return full_weights

# this is deprecated and slow
def average_hankel(hankel,stage,rp,level=None):
    """Averages elements in matrix according to the enforced hankel shape

    Args:
        hankel
        stage
        rp
    Returns:
        hankel_avg
    """
    def _get_level(rp):
        """get level count of multilevel hankel"""
        if rp['recontype'] in ['kx-ky','kx-ky_angio','k-t']:
            return 2

    def _calc_fiber_shape(stage,rp_recontype,rp_fiber_shape):
        """Return fiber shape at current stage as tuple"""

        if rp_recontype in ['kx-ky','kx-ky_angio']:
            (rcvrs,x,y) = rp_fiber_shape
            return (rcvrs,x//2**stage,y//2**stage)
        elif rp_recontype == 'k-t':
            (rcvrs,x,t) = rp_fiber_shape
            return (rcvrs,x//2**stage,t)
    
    def _make_kernel(hankel_shape):
        """makes convolution kernel to sum anti-diagonals"""

        return np.fliplr(np.eye(hankel_shape[1]*2-1))

    def _make_weights(hankel_shape):

        (m,n) = hankel_shape
        weights = np.array([[min(n,i+j+1) for j in range(n)] for i in range(m)])
        weights = np.minimum(weights,np.flipud(np.fliplr(weights)))
        return weights

    def _avg_hankel_block(hankel_block,kernel,weights):
    
        hankel_block = convolve(hankel_block,kernel,mode='constant',cval=0)
        return hankel_block / weights

    if level == None:
        level = _get_level(rp)

    if level == 1:
        raise(Exception('not implemented'))
    elif level == 2:

        (rcvrs,m,n) = _calc_fiber_shape(stage,rp['recontype'],rp['fiber_shape'])
        (p,q) = rp['filter_size']
        hankel_block_shape = (m-p+1,p)
        # make filter for inner block anti-diagonal averaging
        kernel = _make_kernel(hankel_block_shape)
        weights = _make_weights(hankel_block_shape)
        #for allhankelblockswithgenerator????
        # reshape hankel to 4d
        shape4d = (hankel.shape[0]//hankel_block_shape[0],hankel_block_shape[0],\
                    hankel.shape[1]//hankel_block_shape[1],hankel_block_shape[1])
        hankel4d = np.reshape(hankel,shape4d)
        for i in range(hankel4d.shape[0]):
            for j in range(hankel4d.shape[2]):

                hankel4d[i,:,j,:] = _avg_hankel_block(hankel4d[i,:,j,:],\
                                                    kernel, weights)

        hankel = np.reshape(hankel4d,hankel.shape)
        return hankel
                
                
    elif level == 3:
        raise('not implemented')

def init_kspace_stage(kspace_fiber,stage,rp):
    """Setup kspace for current pyramidal stage

    At each stage except 0, kspace data is taken from previos completed stage.
    In wavelet sparse images the pyramidal decomposition means taking the
    center part of the kspace, the size corresponds to the current stage.
   
    Args:
        kspace_fiber (np.ndarray) --  full kspace fiber 
        stage (int) -- stage number, first stage is 0
        rp (dict) -- recon parameters
    Return:
        kspace_init (np.ndarray) --  
    """
    if stage == 0:
        return kspace_fiber
         
    if rp['recontype'] in ['kx-ky','kx-ky_angio']: 

        kx_ind = kspace_fiber.shape[1]
        ky_ind = kspace_fiber.shape[2]
        kspace_init = kspace_fiber[:,kx_ind//2-kx_ind//2**(stage+1): \
                                    kx_ind//2+kx_ind//2**(stage+1),\
                                    ky_ind//2-ky_ind//2**(stage+1):\
                                    ky_ind//2+ky_ind//2**(stage+1)] 
    elif rp['recontype'] == 'k-t':
        kx_ind = kspace_fiber.shape[1]
        kspace_init = kspace_fiber[:,kx_ind//2-kx_ind//2**(stage+1): \
                                    kx_ind//2+kx_ind//2**(stage+1),:] 
        
    elif rp['recontype'] == 'k':

        kx_ind = kspace_fiber.shape[1]
        kspace_init = kspace_fiber[:,kx_ind//2-kx_ind//2**(stage+1): \
                                    kx_ind//2+kx_ind//2**(stage+1)] 

    else:
        raise(Exception('not implemented'))

    return kspace_init

def finish_kspace_stage(kspace_stage, kspace_full, stage_iter,rp):
    """Complete pyramidal stage, put inferred data back into original position

    TODO : maybe put known center data back here
    currently 4 center lines ar placed back
    in kxky a rectangle is used instead of circle

    Args:
        kspace_stage
        kspace_full
        stage_iter (0 or 1) -- indicator: kx-ky needs 2 runs for one stage
        rp (dict) -- recon parameters
    Return:
        kspace_full
    """
    
    if rp['recontype'] in ['kx-ky','kx-ky_angio']:
         
        pe_dim = 1  #TODO dimensions are not dynamic currently
        pe2_dim = 2
        # center coordinates for putting bakc original center data
        center_ind_pe = kspace_full.shape[pe_dim]//2-1
        center_ind_pe2 = kspace_full.shape[pe2_dim]//2-1
        # coordinates for stage size
        ind_start = kspace_full.shape[pe_dim]//2 - kspace_stage.shape[pe_dim]//2
        ind_end = kspace_full.shape[pe_dim]//2 + kspace_stage.shape[pe_dim]//2
        ind2_start = kspace_full.shape[pe2_dim]//2 - kspace_stage.shape[pe2_dim]//2
        ind2_end = kspace_full.shape[pe2_dim]//2 + kspace_stage.shape[pe2_dim]//2
        # saving original center
        center = copy.deepcopy(kspace_full[:,center_ind_pe-2:center_ind_pe+2,\
                                center_ind_pe2-2:center_ind_pe2+2])
        # putting back known center
        kspace_full[:,ind_start:ind_end,ind2_start:ind2_end] = kspace_stage

        kspace_full[:,center_ind_pe-2:center_ind_pe+2,\
                        center_ind_pe2-2:center_ind_pe2+2] = center
        return kspace_full

    elif rp['recontype'] == 'k-t':

        pe_dim = 1  #TODO dimensions are not dynamic currently
        center_ind_pe = kspace_full.shape[pe_dim]//2-1

        ind_start = kspace_full.shape[pe_dim]//2 - kspace_stage.shape[pe_dim]//2
        ind_end = kspace_full.shape[pe_dim]//2 + kspace_stage.shape[pe_dim]//2
        #original center
        center = copy.deepcopy(kspace_full[:,center_ind_pe-2:center_ind_pe+2,:])
        kspace_full[:,ind_start:ind_end,:] = kspace_stage
        kspace_full[:,center_ind_pe-2:center_ind_pe+2,:] = center

        return kspace_full
    
    elif rp['recontype'] == 'k':

        pe_dim = 1  #TODO dimensions are not dynamic currently
        center_ind_pe = kspace_full.shape[pe_dim]//2-1

        ind_start = kspace_full.shape[pe_dim]//2 - kspace_stage.shape[pe_dim]//2
        ind_end = kspace_full.shape[pe_dim]//2 + kspace_stage.shape[pe_dim]//2
        #original center
        center = copy.deepcopy(kspace_full[:,center_ind_pe-2:center_ind_pe+2])
        kspace_full[:,ind_start:ind_end] = kspace_stage
        kspace_full[:,center_ind_pe-2:center_ind_pe+2] = center

        return kspace_full
    else:
        raise(Exception('not implemented yet'))

def make_kspace_weights(rp):
    """Create weight data for kspace weighting

    In the cases containing kx-ky, the weights are applied sequentially in
    the pyramidal decomposition at each stage in the 2 dimensions, thus the 
    returned weight list is twice the length.

    Args:
        rp (dictionary) -- recon parameters
    Return:
        weight_list (list of np.ndarray) -- list of kspace weights according
                                            to pyramidal stages
    """
    def _tv_weights(w):

        if w != 0:
            we = 1j*w
        else:
            we = 0.000001
        return we

    def _haar_weights(s,w):
        """Return weights for Haar wavelets at stage s, frequence w"""

        if w!=0:
            we = 1/np.sqrt(2**s)*(-1j*2**s*w)/2*\
                (np.sin(2**s*w/4)/(2**s*w/4))**2*np.exp(-1j*2**s*w/2)
        else:
            we = 0.000001  # just something small not to divide by 0 accidently
        return we

    # init for all cases
    weight_list = []
    if rp['virtualcoilboost'] == True:
        rcvrs_eff = rp['rcvrs']*2
    else:
        rcvrs_eff = rp['rcvrs']
    
    # generate for each different case
    if rp['recontype'] == 'k':

        kx_len = rp['fiber_shape'][1]
        for s in range(rp['stages']):
            w_samples = [2*np.pi/kx_len*k for k in\
                         range(-int(kx_len/2**(s+1)),int(kx_len/2**(s+1)))]
            w_arr = np.array([_haar_weights(s,i) for i in w_samples],\
                                                            dtype='complex64')
            w_arr = w_arr[np.newaxis,:]
            weight_list.append(w_arr)

        return weight_list

    elif rp['recontype'] == 'k-t':

        kx_len = rp['fiber_shape'][1]
        kt_len = rp['fiber_shape'][2]
        for s in range(rp['stages']):
            w_samples = [2*np.pi/kx_len*k for k in\
                         range(-int(kx_len/2**(s+1)),int(kx_len/2**(s+1)))]
            w_arr = np.array([_haar_weights(s,i) for i in w_samples],\
                                                            dtype='complex64')
            w_arr = w_arr[np.newaxis,:,np.newaxis]
            w_arr = np.repeat(w_arr,rcvrs_eff,axis=0)
            w_arr = np.repeat(w_arr,kt_len,axis=2)
            weight_list.append(w_arr)

        return weight_list

    elif rp['recontype'] == 'kx-ky':

        kx_len = rp['fiber_shape'][1]
        ky_len = rp['fiber_shape'][2]
        for s in range(rp['stages']):
            
            # make nd.array for dim x
            w_samples = [2*np.pi/kx_len*k for k in\
                         range(-int(kx_len/2**(s+1)),int(kx_len/2**(s+1)))]
            w_arr = np.array([_haar_weights(s,i) for i in w_samples],\
                                                            dtype='complex64')

            w_arr = w_arr[np.newaxis,:,np.newaxis]
            w_arr = np.repeat(w_arr,rcvrs_eff,axis=0)
            w_arr = np.repeat(w_arr,ky_len//2**s,axis=2)
            weight_list.append(w_arr)
            # make nd.array for dim x
            w_samples = [2*np.pi/ky_len*k for k in\
                         range(-int(ky_len/2**(s+1)),int(ky_len/2**(s+1)))]
            w_arr = np.array([_haar_weights(s,i) for i in w_samples],\
                                                            dtype='complex64')
            w_arr = w_arr[np.newaxis,np.newaxis,:]
            w_arr = np.repeat(w_arr,rcvrs_eff,axis=0)
            w_arr = np.repeat(w_arr,kx_len//2**s,axis=1)
            weight_list.append(w_arr)

        return weight_list

    elif rp['recontype'] == 'kx-ky_angio':
        
        kx_len = rp['fiber_shape'][1]
        ky_len = rp['fiber_shape'][2]
        
        # make nd.array for dim x
        w_samples = [2*np.pi/kx_len*k for k in\
                     range(-int(kx_len/2),int(kx_len/2))]
        w_arr = np.array([_tv_weights(i) for i in w_samples],\
                                                        dtype='complex64')

        w_arr = w_arr[np.newaxis,:,np.newaxis]
        w_arr = np.repeat(w_arr,rcvrs_eff,axis=0)
        w_arr = np.repeat(w_arr,ky_len,axis=2)
        weight_list.append(w_arr)
        # make nd.array for dim x
        w_samples = [2*np.pi/ky_len*k for k in\
                     range(-int(ky_len/2),int(ky_len/2))]
        w_arr = np.array([_tv_weights(i) for i in w_samples],\
                                                        dtype='complex64')
        w_arr = w_arr[np.newaxis,np.newaxis,:]
        w_arr = np.repeat(w_arr,rcvrs_eff,axis=0)
        w_arr = np.repeat(w_arr,kx_len,axis=1)
        weight_list.append(w_arr)

        return weight_list

    elif rp['recontype'] == 'kx-ky-t':
        raise(Exception('Not implemented yet!'))

def apply_kspace_weights(kspace_fiber,weight):
    """Mutiply n-D kspace elements with the approppriate weights

    K-space weighing is a major part in Aloha framework. This function
    determines the proper weighting type, creates and applies the weighting.
    "fiber" is the n-dimensional type of a slice from the k-space.
    Example: in k-t case, the fiber is a 3d cut from kspace along the
    [rcvrs,pe,te] dimensions.

    It!s just a multiplication though, maybe error check...

    Args:
        fiber (np.ndarray) -- input kspace part to weigh
        rp (dictionary) -- recon parameter dictionary
    Return:
        fiber_weighted
    """
    try:
        weighted = np.multiply(kspace_fiber,weight) 
    except:
        raise
    
    return weighted 

def remove_kspace_weights(kspace_fiber,weight):
    """Just divide kspace fiber by the weights"""

    try:
        unweighted = np.divide(kspace_fiber,weight) 
    except:
        raise
    
    return unweighted 

def construct_hankel(nd_data, rp, level=None,order=None):
    """Contruct Multilevel Hankel matrix

    Args:
        nd_data : (np.ndarray) -- input n dimensional data. dim 0 is assumed
                                to be the receiver dimension
        rp (dictionary) -- vnmrjpy aloha reconpar dictionary
        order (list) -- multileveling order. list elements are the dimensions
        level (int) -- number of levels (if None it is assumed from rp)

    Returns:
        hankel (np.ndarray(x,y)) 2D multilevel Hankel matrix
    """
    def _construct_hankel_level(block_vector, filt):
        """Create hanekl from a vector of matrices

        Args:
            block_vector (np.ndarray) --  vector of matrices
            filt (int)  -- filter size on dimension according to Hankel level
        Return:
            hankel
        """
        if len(block_vector.shape) == 1:
            lvl=1
            block_vector = np.expand_dims(block_vector,axis=-1) 
            block_vector = np.expand_dims(block_vector,axis=-1) 
        if len(block_vector.shape) == 3:
            lvl=2
        else:
            raise(Exception('weird shape'))
        shape_x = (block_vector.shape[0]-filt+1)*block_vector.shape[1]
        shape_y = block_vector.shape[2]*filt
        hankel = np.zeros((shape_x,shape_y),dtype='complex64')

        row_num = block_vector.shape[0]-filt+1
        col_num = filt
        row_size = block_vector.shape[1]
        col_size = block_vector.shape[2]
        for col in range(col_num):

            for row in range(row_num):

                hankel[row*row_size : (row+1)*row_size,\
                        col*col_size : (col+1)*col_size] = \
                                            block_vector[col+row,:,:]

        return hankel

    nd_data = np.array(nd_data, dtype='complex64')

    # getting level
    if level == None: 
        level = len(rp['filter_size'])

    if level == 1:

        raise(Exception('not implemented'))

    if level == 2:    
        # annihilating filter size
        p, q = rp['filter_size']
        # hankel m n
        m, n = nd_data.shape[1:]
        #init final shape
        shape_x = (m-p+1)*(n-q+1)
        shape_y = p*q*nd_data.shape[0]  # concatenate along receiver dimension
        shape_x_lvl1 = m-p+1
        shape_y_lvl1 = p
        shape_x_rcvr = shape_x
        shape_y_rcvr = shape_y//nd_data.shape[0]
        
        hankel_rcvr = np.zeros((shape_x_rcvr,shape_y_rcvr),dtype='complex64')
        hankel_lvl1_vec = np.zeros((n,shape_x_lvl1,shape_y_lvl1),dtype='complex64')
        hankel = np.zeros((shape_x,shape_y),dtype='complex64')

        for rcvr in range(nd_data.shape[0]):

            for j in range(nd_data.shape[2]):
               
                #vec = [nd_data[rcvr,i,j] for i in range(nd_data.shape[1])]
                #vec = np.array(vec)
                vec = nd_data[rcvr,:,j]
                hankel_lvl1 = _construct_hankel_level(vec,p)
                hankel_lvl1_vec[j,:,:] = hankel_lvl1
            hankel_rcvr = _construct_hankel_level(hankel_lvl1_vec,q)
            hankel[:,rcvr*shape_y_rcvr:(rcvr+1)*shape_y_rcvr] = hankel_rcvr

        return hankel

    if level == 3:
        raise(Exception('not implemented'))

def deconstruct_hankel(hankel,stage,rp):
    """Make the original ndarray from the multilevel Hankel matrix.

    Used upon completion, and also in an ADMM averaging step
    
    Args:
        hankel (np.ndarray) -- input hankel matrix
        rp (dictionary) -- recon parameters
        stage (int) -- current pyramidal stage
    Return:
        nd_data (np.ndarray)

    """
    def _calc_fiber_shape(stage,rp_recontype,rp_fiber_shape):
        """Return fiber shape at current stage as tuple"""

        if rp_recontype in ['kx-ky','kx-ky_angio']:
            (rcvrs,x,y) = rp_fiber_shape
            return (rcvrs,x//2**stage,y//2**stage)
        elif rp_recontype == 'k-t':
            (rcvrs,x,t) = rp_fiber_shape
            return (rcvrs,x//2**stage,t)
        elif rp_recontype == 'k':
            (rcvrs,x) = rp_fiber_shape
            return (rcvrs,x//2**stage)
        else:
            raise(Exception('not implemented'))
    
    def _block_vector_from_col(col, height):
        """Make vector of blocks from a higher level Hankel column"""

        block_num = col.shape[0]//height
        block_x = height
        block_y = col.shape[1]
        vec = np.zeros((block_num,block_x,block_y),dtype='complex64')
        for num in range(block_num):
            vec[num,:,:] = col[num*block_x:(1+num)*block_x,:]
        return vec

    def _deconstruct_hankel_level(hankel,m,p,factors):
        """ Make an array of Hankel building block

        This is the main local function         
        
        Args:
            hankel -- 2 dim array to decompose
            blockshape -- tuple, shape of inner matrices building up the Hankel
        Return:
            vec -- 3 dim array, dim 0 counts the individual blocks
        """
        inner_x = hankel.shape[0]//(m-p+1)
        inner_y = hankel.shape[1]//p
        block_vector = np.zeros((m,inner_x,inner_y),dtype='complex64')
        for col in range(p):
            to_add = np.zeros((m,inner_x,inner_y),dtype='complex64')
            hankel_col = hankel[:,col*inner_y:(col+1)*inner_y]
            vec = _block_vector_from_col(hankel_col,inner_x)
            to_add[col:col+vec.shape[0],:,:] = vec
            block_vector = block_vector + to_add
        # average the lower level hankel blocks 
        hankels_inner = _average_block_vector(block_vector,factors) 
        return hankels_inner

    def _average_block_vector(vec,factors):
        """Return list of lover level hankels"""

        if len(factors.shape) != len(vec.shape):
            factors = factors[:,np.newaxis,np.newaxis]
        hankels = np.divide(vec,factors)
        return hankels

    def _make_factors(n,q):
        """Make array of Hankel block multiplicities for averaging

        Args:
            n -- kspace length in a dimension
            q -- annihilating filter length in same dimension
        Return:
            factors -- np.array of lenth n
        """
        factors = np.zeros(n) # multiplicity of 1 block
        ramp = [i for i in range(1,q)]
        factors[0:len(ramp)] = ramp
        factors[n-len(ramp):] = ramp[::-1]
        factors[factors == 0] = q
        return factors

    # init
    if rp['recontype'] in ['kx-ky','k-t','kx-ky_angio']:
        level = 2
    elif rp['recontype'] in ['k']:
        level = 1
    # work work 
    if level == 1:

        # subinit level
        (rcvrs,m) = _calc_fiber_shape(stage,rp['recontype'],rp['fiber_shape'])
        p = rp['filter_size']  # annihilating filter
        #number of hankel hankel block columns
        cols_rcvr_lvl2 = hankel.shape[1]//(rcvrs*p) 
        # zeroinit final output
        nd_data = np.zeros((rcvrs,m),dtype='complex64')        

        factors_lvl1 = _make_factors(m,p) # multiplicity of 1 block
        
        # decompose level2
        for rcvr in range(rcvrs):

            hankel_rcvr = hankel[:,rcvr*p:(rcvr+1)*p]
            # decompose level1
            elements = _deconstruct_hankel_level(hankel_rcvr,m,p,factors_lvl1)
            nd_data[rcvr,:] = elements[:,0,0]  # dim 1,2 are single

        #print('nd data shape {}'.format(nd_data.shape))
        return nd_data

    elif level == 2:
        
        # subinit level
        (rcvrs,m,n) = _calc_fiber_shape(stage,rp['recontype'],rp['fiber_shape'])
        (p,q) = rp['filter_size']  # annihilating filter
        #number of hankel hankel block columns
        cols_rcvr_lvl2 = hankel.shape[1]//(rcvrs*p) 
        # zeroinit final output
        nd_data = np.zeros((rcvrs,m,n),dtype='complex64')        

        factors_lvl2 = _make_factors(n,q) # multiplicity of 1 block
        factors_lvl1 = _make_factors(m,p) # multiplicity of 1 block
        
        # decompose level2
        for rcvr in range(rcvrs):

            hankel_rcvr = hankel[:,rcvr*p*q:(rcvr+1)*p*q]
            hankels_lvl1 = _deconstruct_hankel_level(hankel_rcvr,n,q,\
                                                            factors_lvl2)
            # decompose level1
            for num in range(hankels_lvl1.shape[0]):
                hank = hankels_lvl1[num,:,:]
                elements = _deconstruct_hankel_level(hank,m,p,factors_lvl1)
                nd_data[rcvr,:,num] = elements[:,0,0]  # dim 1,2 are single

        return nd_data

    elif level == 3:
        raise(Exception('not implemented'))
    else:
        raise(Exception('not implemented'))

