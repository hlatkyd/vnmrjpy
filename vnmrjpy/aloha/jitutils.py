import numpy as np
import numba


@numba.njit
def avg_lvl1_hankel(hankel_full,rcvrs):
    """Average matrix, assuming hankel structure
    
    Args:
        hankel (np.ndarray) -- matrix to average
        rcvrs (int) -- receiver channels
    Returns:
        hankel (np.ndarray) -- finished matrix
    """
    a = hankel_full.shape[1]//rcvrs
    for rcvr in range(rcvrs):

        hankel = hankel_full[:,rcvr*a:(rcvr+1)*a]

        num_antidiag = hankel.shape[0]+hankel.shape[1]-1
        # get indice pairs of each antidiagonal
        for ind in range(num_antidiag):
            antidiag_pairs = [[ind-j,j] for j in range(ind+1) \
                            if (j<hankel.shape[1] and ind-j < hankel.shape[0])]
            # avegage antidiagonals
            avg = 0
            for pair in antidiag_pairs:
                avg += hankel[pair[0],pair[1]]
            avg = avg / len(antidiag_pairs)
            # putting back
            for pair in antidiag_pairs:
                hankel[pair[0],pair[1]] = avg

        hankel_full[:,rcvr*a:(rcvr+1)*a]

    return hankel_full

@numba.njit
def avg_lvl2_hankel(hankel_full, block_shape, rcvrs):
    """Averages level 2 block hankel matrix

    Args:
        hankel (2D np.ndarray) -- input matrix to average
        block_shape (tuple) -- inner block shape
        rcvrs (int) -- receiver channels
    Returns:
        hankel (2D np.ndarray) -- averaged matrix

    """

    # -----------------init-----------------
    # outer_shape
    (a,b) = (hankel_full.shape[0]//block_shape[0],\
                hankel_full.shape[1]//block_shape[1]//rcvrs)
    (c,d) = block_shape
    # get indice paris for inner block antidiagonals
    pairs_inner = []
    for ind in range(block_shape[0]+block_shape[1]-1):
        pairs_inner.append([[ind-j,j] for j in range(ind+1) \
                        if (j<d and ind-j < c)])
    pairs_outer = []
    for ind in range(a+b-1):
        pairs_outer.append([[ind-j,j] for j in range(ind+1) \
                        if (j<b and ind-j < a)])
    # -------------actual calc--------------
    # iterate over receiver concatenated blocks
    for rcvr in range(rcvrs):

        hankel = hankel_full[:,rcvr*b*d:(rcvr+1)*b*d]

        # iterate over each block antidiagonal
        for ind in range(a+b-1):
            
            # selecting blocks, make a list
            ij_list = pairs_outer[ind]
            block_list = [hankel[ij[0]*c:(ij[0]+1)*c,ij[1]*d:(ij[1]+1)*d] \
                                                for ij in ij_list]
            # now average the blocks
            temp = block_list[0]
            for k in range(1,len(block_list)):
                temp = np.add(temp,block_list[k])
            avgblock = temp / len(block_list)
            #average antidiag in avgblock
            for antidiag in range(c+d-1):
                avg = 0
                for pair in pairs_inner[antidiag]:
                    avg += avgblock[pair[0],pair[1]]

                avg = avg / len(pairs_inner[antidiag])
                for pair in pairs_inner[antidiag]:
                    avgblock[pair[0],pair[1]] = avg

            # average block is ready, putting bak into lvl2 hankel
            for ij in ij_list:
                hankel[ij[0]*c:(ij[0]+1)*c,ij[1]*d:(ij[1]+1)*d] = avgblock
             
        hankel_full[:,rcvr*b*d:(rcvr+1)*b*d]

    return hankel_full

@numba.njit(parallel=True)
def _hankel_lvl1(block_vector, filt):
    """Create inner (level1) hankel from a vector

    Args:
        block_vector (np.ndarray) --  vector of matrices
        filt (int)  -- filter size on dimension according to Hankel level
    Return:
        hankel
    """
    block_vector = np.expand_dims(block_vector,axis=-1) 
    block_vector = np.expand_dims(block_vector,axis=-1) 
    
    shape_x = (block_vector.shape[0]-filt+1)*block_vector.shape[1]
    shape_y = block_vector.shape[2]*filt
    hankel = np.zeros((shape_x,shape_y),numba.complex64)

    row_num = block_vector.shape[0]-filt+1
    col_num = filt
    row_size = block_vector.shape[1]
    col_size = block_vector.shape[2]
    for col in numba.prange(col_num):

        for row in range(row_num):

            hankel[row*row_size : (row+1)*row_size,\
                    col*col_size : (col+1)*col_size] = \
                                        block_vector[col+row,:,:]

    return hankel

@numba.njit(parallel=True)
def _hankel_lvl2(block_vector, filt):
    """Create level2 hankel from a vector of matrices

    Args:
        block_vector (np.ndarray) --  vector of matrices
        filt (int)  -- filter size on dimension according to Hankel level
    Return:
        hankel
    """
    shape_x = (block_vector.shape[0]-filt+1)*block_vector.shape[1]
    shape_y = block_vector.shape[2]*filt
    hankel = np.zeros((shape_x,shape_y),numba.complex64)

    row_num = block_vector.shape[0]-filt+1
    col_num = filt
    row_size = block_vector.shape[1]
    col_size = block_vector.shape[2]
    for col in numba.prange(col_num):

        for row in range(row_num):

            hankel[row*row_size : (row+1)*row_size,\
                    col*col_size : (col+1)*col_size] = \
                                        block_vector[col+row,:,:]

    return hankel

@numba.njit()
def construct_lvl1_hankel(nd_data, filter_size):
    """Contruct Level1 Hankel matrix

    Args:
        nd_data : (np.ndarray) -- input n dimensional data. dim 0 is assumed
                                to be the receiver dimension
        filter_size (int) -- annihilating filter size

    Returns:
        hankel (np.ndarray(x,y)) 2D multilevel Hankel matrix
    """
    # annihilating filter size
    p = filter_size
    # hankel m n
    m = nd_data.shape[1:][0]
    #init final shape
    shape_x = m-p+1
    shape_y = p*nd_data.shape[0]  # concatenate along receiver dimension
    shape_x_lvl1 = m-p+1
    shape_y_lvl1 = p
    shape_x_rcvr = shape_x
    shape_y_rcvr = shape_y//nd_data.shape[0]
   
    #hankel_rcvr = np.zeros((shape_x_rcvr,shape_y_rcvr),'complex64')
    #hankel = np.zeros((shape_x,shape_y),'complex64')

    hankel_rcvr = np.zeros((shape_x_rcvr,shape_y_rcvr),numba.complex64)
    hankel = np.zeros((shape_x,shape_y),numba.complex64)
    
    for rcvr in range(nd_data.shape[0]):

        vec = nd_data[rcvr,:]
        hankel_rcvr = _hankel_lvl1(vec,p)

        hankel[:,rcvr*shape_y_rcvr:(rcvr+1)*shape_y_rcvr] = hankel_rcvr

    return hankel

@numba.njit()
def construct_lvl2_hankel(nd_data, filter_size):
    """Contruct Multilevel Hankel matrix

    Args:
        nd_data : (np.ndarray) -- input n dimensional data. dim 0 is assumed
                                to be the receiver dimension
        rp (dictionary) -- vnmrjpy aloha reconpar dictionary

    Returns:
        hankel (np.ndarray(x,y)) 2D multilevel Hankel matrix
    """
    # annihilating filter size
    p, q = filter_size
    # hankel m n
    m, n = nd_data.shape[1:]
    #init final shape
    shape_x = (m-p+1)*(n-q+1)
    shape_y = p*q*nd_data.shape[0]  # concatenate along receiver dimension
    shape_x_lvl1 = m-p+1
    shape_y_lvl1 = p
    shape_x_rcvr = shape_x
    shape_y_rcvr = shape_y//nd_data.shape[0]
   
    hankel_rcvr = np.zeros((shape_x_rcvr,shape_y_rcvr),numba.complex64)
    hankel_lvl1_vec = np.zeros((n,shape_x_lvl1,shape_y_lvl1),numba.complex64)
    hankel = np.zeros((shape_x,shape_y),numba.complex64)
    
    for rcvr in range(nd_data.shape[0]):

        for j in range(nd_data.shape[2]):
           
            #vec = [nd_data[rcvr,i,j] for i in range(nd_data.shape[1])]
            #vec = np.array(vec)
            vec = nd_data[rcvr,:,j]
            hankel_lvl1 = _hankel_lvl1(vec,p)
            hankel_lvl1_vec[j,:,:] = hankel_lvl1
        hankel_rcvr = _hankel_lvl2(hankel_lvl1_vec,q)
        hankel[:,rcvr*shape_y_rcvr:(rcvr+1)*shape_y_rcvr] = hankel_rcvr

    return hankel
#---------------------------------SLOOWWWWWWWW----------------------------
@numba.njit()
def deconstruct_lvl2_hankel(hankel,stage,recontype_int,fiber_shape,filter_size):
    """Make the original ndarray from the multilevel Hankel matrix.

    Used upon completion, and also in an ADMM averaging step
    
    Args:
        hankel (np.ndarray) -- input hankel matrix
        rp (dictionary) -- recon parameters
        stage (int) -- current pyramidal stage
    Return:
        nd_data (np.ndarray)

    """
    
    if recontype_int == 1:
        (rcvrs,x,y) = fiber_shape
        (rcvrs,m,n) = (rcvrs,x//2**stage,y//2**stage)
    elif recontype_int == 2:
        (rcvrs,x,y) = fiber_shape
        (rcvrs,m,n) = (rcvrs,x//2**stage,y)
    (p,q) = filter_size  # annihilating filter
    #number of hankel hankel block columns
    cols_rcvr_lvl2 = hankel.shape[1]//(rcvrs*p) 
    # zeroinit final output
    nd_data = np.zeros((rcvrs,m,n),numba.complex64)        

    factors_lvl2 = _make_factors(n,q) # multiplicity of 1 block
    factors_lvl1 = _make_factors(m,p) # multiplicity of 1 block
    
    # decompose level2
    for rcvr in range(rcvrs):

        hankel_rcvr = hankel[:,rcvr*p*q:(rcvr+1)*p*q]
        hankels_lvl1 = _deconstruct_hankel_lvl2(hankel_rcvr,n,q,factors_lvl2)
        # decompose level1
        for num in range(hankels_lvl1.shape[0]):
            hank = hankels_lvl1[num,:,:]
            elements = _deconstruct_hankel_lvl1(hank,m,p,factors_lvl1)
            nd_data[rcvr,:,num] = elements[:,0,0]  # dim 1,2 are single

    return nd_data

@numba.njit()
def _calc_fiber_shape(stage,recontype_int,fiber_shape):
    """Return fiber shape at current stage as tuple"""

    if recontype_int == 1:
        (rcvrs,x,y) = fiber_shape
        return (rcvrs,x//2**stage,y//2**stage)
    elif recontype_int == 2:
        (rcvrs,x,t) = fiber_shape
        return (rcvrs,x//2**stage,t)

@numba.njit()
def _block_vector_from_col(col, height):
    """Make vector of blocks from a higher level Hankel column"""

    block_num = col.shape[0]//height
    block_x = height
    block_y = col.shape[1]
    vec = np.zeros((block_num,block_x,block_y),numba.complex64)
    for num in range(block_num):
        vec[num,:,:] = col[num*block_x:(1+num)*block_x,:]
    return vec

@numba.njit()
def _deconstruct_hankel_lvl1(hankel, m, p, factors):
    """ Make an array of Hankel building block

    This is the main local function         
    
    Args:
        hankel -- 2 dim array to decompose
        blockshape -- tuple, shape of inner matrices building up the Hankel
        level (int) -- specifies hankel level 
    Return:
        vec -- 3 dim array, dim 0 counts the individual blocks
    """
    inner_x = hankel.shape[0]//(m-p+1)
    inner_y = hankel.shape[1]//p
    block_vector = np.zeros((m,inner_x,inner_y),numba.complex64)
    for col in range(p):
        to_add = np.zeros((m,inner_x,inner_y),numba.complex64)
        hankel_col = hankel[:,col*inner_y:(col+1)*inner_y]
        vec = _block_vector_from_col(hankel_col,inner_x)
        to_add[col:col+vec.shape[0],:,:] = vec
        block_vector = block_vector + to_add
        # average the lower level hankel blocks
        hankels_inner = np.divide(block_vector,factors)
    return hankels_inner

@numba.njit()
def _deconstruct_hankel_lvl2(hankel, m, p, factors):
    """ Make an array of Hankel building block

    This is the main local function         
    
    Args:
        hankel -- 2 dim array to decompose
        blockshape -- tuple, shape of inner matrices building up the Hankel
        level (int) -- specifies hankel level 
    Return:
        vec -- 3 dim array, dim 0 counts the individual blocks
    """
    inner_x = hankel.shape[0]//(m-p+1)
    inner_y = hankel.shape[1]//p
    block_vector = np.zeros((m,inner_x,inner_y),numba.complex64)
    for col in range(p):
        to_add = np.zeros((m,inner_x,inner_y),numba.complex64)
        hankel_col = hankel[:,col*inner_y:(col+1)*inner_y]
        vec = _block_vector_from_col(hankel_col,inner_x)
        to_add[col:col+vec.shape[0],:,:] = vec
        block_vector = block_vector + to_add
        # average the lower level hankel blocks
        _factors = np.expand_dims(np.expand_dims(factors,axis=-1),axis=-1)
        hankels_inner = np.divide(block_vector,_factors)
    return hankels_inner

@numba.njit()
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

