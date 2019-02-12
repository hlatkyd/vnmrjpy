import copy
import numpy as np
import vnmrjpy as vj
import sys
import numba

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
"""
Collection of solvers for low rank matrix completion
"""
def svt(A,known_data=None,tau=None, delta=None, epsilon=1e-4,max_iter=1000,\
        realtimeplot=False):
    """Low rank matrix completion algorihm with singular value thresholding

    Uses singular value decomposition then throws away the small singular
    values each iteration. Defaults are used as in the reference paper

    Ref.: Candes paper

    Args:
        A (np.ndarray) -- 2 dimensional numpy array to be completed
        known_data (np.ndarray)-- known elements of A in the same shape,
                                    unknown are 0
        tau (float)
        delta (float)
        epsilon (float)
        max_iter (int) -- maximum number of iterations
        realtimeplot (boolean) -- convenience option to plot the result each 
                                iteration
    Return:
        A_filled (np.ndarray) -- completed 2dim numpy array
    """
    Y = np.zeros_like(A)
    mask = np.zeros_like(A)
    mask[known_data != 0] = 1

    if tau == None:
        tau = 5*np.sum(A.shape)/2
    if delta == None:
        delta = 1.2*np.prod(A.shape)/np.sum(mask)
    if realtimeplot == True:
        rtplot = vj.util.RealTimeImshow(np.abs(A))
    
    # start iteration
   
    for _ in range(max_iter): 
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        S = np.maximum(S-tau, 0)
        X = np.linalg.multi_dot([U, np.diag(S), V])
        Y = Y + delta*mask*(A-X)
    
        rel_recon_error = np.linalg.norm(mask*(X-A)) / \
                            np.linalg.norm(mask*A)

        if _ % 100 == 0:
            sys.stdout.flush()
            print(rel_recon_error)
            pass
        if rel_recon_error < epsilon:
            break
        if realtimeplot == True:
            rtplot.update_data(np.absolute(X))

    return X


def lmafit(init_data,known_data=None,\
            tol=5e-4,\
            k=1,\
            max_iter=100,\
            rank_strategy='increase',\
            verbose=False,\
            realtimeplot=False):
    """Low-rank matrix fitting algorithm

    Fills missing matrix elements by low rank approximation

    ref.: paper

    Args:
        init_data (np.ndarray) : matrix to complete. unkown elements can
                be approximated beforehand
        known_data (np.ndarray) : matrix of the same shape as init_data 
                with only the known elements, rest are zero
        tol (float) : tolerance of fitting
        rank_strategy (str) : increase or decrease rank
        verbose
        realtimeplot

    Returns:
        X (np.matrix)
        Y (np.matrix) 
        out_list [] : some list of helper outputs
    """
    (m,n) = init_data.shape
    datamask = np.zeros_like(known_data)
    datamask[known_data != 0+0*1j] = 1
    datanrm = np.linalg.norm(init_data)
    # init
    #Z = np.matrix(init_data)
    Z = init_data
    #X = np.matrix(np.zeros((m,k),dtype='complex64'))
    #TODO check removed matrix ok
    X = np.zeros((m,k),dtype='complex64')
    Y = np.eye(k,n,dtype='complex64')
    Res = np.multiply(init_data,datamask) - known_data
    res = datanrm
    reschg_tol = 0.5*tol
    # parameters for alf
    alf = 0
    increment = 0.5
    #rank estimation parameters
    itr_rank = 0
    minitr_reduce_rank = 5
    maxitr_reduce_rank = 50
    tau_limit = 10
    rank_incr = 3
    rank_max = 50

    if realtimeplot == True:
        rtplot = vj.util.RealTimeImshow(np.absolute(init_data))

    def rank_check(R,reschg,tol):
        
        # diag = np.diag(R)
        # d_hat = [diag[i]/diag[i+1] for i in range(len(diag)-1)]
        # tau = (len(diag)-1)*max(d_hat)/(sum(d_hat)-max(d_hat))

        if reschg < 10*tol:
            ind_string = 'increase'
        else:
            ind_string = 'stay'
        return ind_string

    def increase_rank(X,Y,Z,rank_incr,rank_max):
        
        k = X.shape[1]
        k_new = min(k+rank_incr,rank_max)

        m = X.shape[0]
        n = Y.shape[1]
        #X_new = np.matrix(np.zeros((m,k_new),dtype='complex64'))
        #Y_new = np.matrix(np.eye(k_new,n,dtype='complex64'))            
        X_new = np.zeros((m,k_new),dtype='complex64')
        Y_new = np.eye(k_new,n,dtype='complex64')          
        X_new[:,:k] = X
        Y_new[:k,:] = Y
        Z_new = X.dot(Y)
        return X_new, Y_new, Z_new

    # --------------MAIN ITERATION--------------------
    objv = np.zeros(max_iter)
    RR = np.ones(max_iter)

    for iter_ in range(max_iter):
        itr_rank += 1
        """
        X0 = copy.deepcopy(X)
        Y0 = copy.deepcopy(Y)
        Res0 = copy.deepcopy(Res)
        res0 = copy.deepcopy(res)
        Z0 = copy.deepcopy(Z)
        """
        X0 = X
        Y0 = Y
        Res0 = Res
        res0 = res
        Z0 = Z
        X = Z.dot(Y.conj().T)
        X, R = np.linalg.qr(X)
        Y = X.conj().T.dot(Z)
        Z = X.dot(Y)

        Res = np.multiply(known_data-Z,datamask)
        res = np.linalg.norm(Res,'fro')
        relres = res / datanrm
        ratio = res / res0
        reschg = np.abs(1-ratio)
        RR[iter_] = ratio
        # adjust alf
        if verbose == True:

            print('ratio : {}; rank : {}; reschg : {}, alf : {}'\
                    .format(ratio,X.shape[1],reschg, alf))

        if ratio >= 1.0:
            increment = np.max([0.1*alf,0.1*increment])
            X = X0
            Y = Y0
            Res =Res0
            res =res0
            relres = res / datanrm
            alf = 0
            Z = Z0
        elif ratio > 0.7:
            increment = max(increment,0.25*alf)
            alf = alf + increment 
        objv[iter_] = relres
        # check stopping
        if ((reschg < reschg_tol) and ((itr_rank > minitr_reduce_rank) \
                                or (relres < tol))):
            if verbose == True:
                print('Stopping crit achieved')
            break

        # rank adjustment
        rankadjust = rank_check(R,reschg,tol)
        if rankadjust == 'increase':
            X,Y,Z = increase_rank(X,Y,Z,rank_incr,rank_max)

        Zknown = known_data + alf*Res
        Z = Z - np.multiply(Z,datamask) + Zknown

        if realtimeplot == True:
            rtplot.update_data(np.absolute(Z))

    obj = objv[:iter_]

    return X, Y, [obj, RR, iter_, relres, reschg] 

def admm(U,V,fiber_stage_known,stage,rp,\
                                    mu=1000,\
                                    max_iter=100,\
                                    realtimeplot=False,\
                                    noiseless=True,\
                                    verbose=False):
    """Alternating Direction Method of Multipliers solver for Aloha


    Tuned for ALOHA MRI reconstruction framework, not for general use yet.
    Lmafit estimates the rank, then Admm is used to enforce the hankel structure

    refs.: Aloha papers ?
            Admm paper ?
        
    Args:
        U, V  (np.matrix) -- U*V.H is the estimated hankel matrix from Lmafit
        fiber_stage_known (np.ndarray) -- known elements in fiber,zerofilled
        stage (int) -- pyramidal decomposition stage ?? WHY??
        rp {dictionary} -- Aloha recon parameters 
        realitmeplot (boolean) -- option to plot each iteration
        noiseless (boolean) -- ?
    Returns:
        hankel
    """

    fiber_shape = fiber_stage_known.shape
    if len(fiber_shape) == 3:  # this is kx-ky, angio and k-t
        (rcvrs,m,n) = fiber_shape
        (p,q) = rp['filter_size']
        hankel_block_shape = (m-p+1,p)
        hankel_level = 2
    elif len(fiber_shape) == 2:  # this is simple k, eg.: gems
        (rcvrs,m) = fiber_shape
        p = rp['filter_size']
        hankel_block_shape = (m-p+1,p)
        hankel_level = 1
    else:
        raise(Exception('not implemented yet'))

    # init construction
    fiber_orig_part = copy.copy(fiber_stage_known)
    if hankel_level == 2:
        hankel_orig_part = vj.aloha.construct_lvl2_hankel(\
                                        fiber_orig_part,rp['filter_size'])
        hankel_mask = np.absolute(vj.aloha.construct_lvl2_hankel(\
                                        fiber_stage_known,rp['filter_size']))
    elif hankel_level == 1:
        hankel_orig_part = vj.aloha.construct_lvl1_hankel(\
                                        fiber_orig_part,rp['filter_size'])
        hankel_mask = np.absolute(vj.aloha.construct_lvl1_hankel(\
                                        fiber_stage_known,rp['filter_size']))
    #hankel_mask = np.absolute(vj.aloha.construct_hankel(fiber_stage_known,rp))
    hankel_mask[hankel_mask != 0] = 1
    hankel_mask = np.array(hankel_mask,dtype='complex64')  # where data is unknown
    hankel_mask_inv = np.ones(hankel_mask.shape) - hankel_mask

    hankel0 = U.dot(V.conj().T)
    hankel = copy.copy(hankel0)
    #hankel = U.dot(V.conj().T)

    # TODO do thios one better
    # init lagrangian update
    lagr = np.zeros(hankel.shape,dtype='complex64')
    #lagr = copy.deepcopy(hankel)
    us = (U.conj().T.dot(U)).shape
    vs = (V.conj().T.dot(V)).shape
    Iu = np.eye(us[0],us[1],dtype='complex64')
    Iv = np.eye(vs[0],vs[0],dtype='complex64')

    if verbose:
        
        res = np.linalg.norm(hankel_mask*U.dot(V.conj().T)-hankel_orig_part)
        print('mean error: {}'.format(res))

    # real time plotting for debugging purposes
    realtimeplot = realtimeplot
    if realtimeplot == True:
        rtplot = vj.util.RealTimeImshow(np.absolute(hankel0))

    for _ in range(max_iter):
    
        #start = time.time()
        # average the hankel structure, and put back original elements
        if hankel_level == 2:
            hankel = vj.aloha.avg_lvl2_hankel(hankel,hankel_block_shape,rcvrs)
        elif hankel_level == 1:
            hankel = vj.aloha.avg_lvl1_hankel(hankel,rcvrs)

        #print('admm hankel shape {}'.format(hankel.shape))
        hankel = np.multiply(hankel,hankel_mask_inv) + hankel_orig_part
        # updating U,V and the lagrangian
        #U_calc_inv = np.linalg.inv(Iv+mu*V.conj().T.dot(V))
        U = mu*(hankel0+lagr).dot(V).dot(\
                            np.linalg.inv(Iv+mu*V.conj().T.dot(V)))
        #V_calc_inv = np.linalg.inv(Iu+mu*U.conj().T.dot(U))
        V = mu*((hankel0+lagr).conj().T).dot(U).dot(\
                            np.linalg.inv(Iu+mu*U.conj().T.dot(U)))
        hankel = U.dot(V.conj().T)
        lagr = hankel0 - hankel + lagr

        if verbose:
            res = np.linalg.norm(hankel_mask*U.dot(V.conj().T)-hankel_orig_part)
            print('mean error: {}'.format(res))
        if realtimeplot == True:
            rtplot.update_data(np.absolute(hankel))
        #print('admm iter time {}'.format(time.time()-start))

    return hankel   
