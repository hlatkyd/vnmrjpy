#!/usr/local/bin/python3

import copy
import numpy as np
import sys
DTYPE = 'complex64'

class ADMM():
    """
    Class for problem solving with Alternating Direction Method of Multipliers
    Used for a low rank matrix completion step after, U,V in H = U*V' have been
    initialized by LMaFit. Used in ALOHA MRI reconstruction framework.
    -------------------------------------------------------------------------
    Input:

    -------------------------------------------------------------------------
    Methods:
        
        solve
        solve_CUDA
    """

    def __init__(self,U,V,slice3d_cs_weighted,slice3d_shape,\
                    stage,rp, realtimeplot=True):

        U = np.matrix(U)
        V = np.matrix(V)
        if rp['recontype'] in ['k-t','kx-ky','kx-ky_angio']:
            self.hankel_mask = np.absolute(compose_hankel_2d(slice3d_cs_weighted,rp))

            self.decomp_factors = make_hankel_decompose_factors(\
                                    slice3d_shape,rp)
        elif rp['reontype'] == 'higher dim':
            raise(Exception('not implemented'))

        self.hankel_mask[self.hankel_mask != 0] = 1
        self.hankel_mask = np.array(self.hankel_mask,dtype=DTYPE)
        self.hankel_mask_inv = np.ones(self.hankel_mask.shape) - self.hankel_mask
        # real time plotting for debugging purposes
        self.realtimeplot = realtimeplot
        if realtimeplot == True:
            self.rtplot = RealTimeImshow(np.absolute(U.dot(V.H)))
        # putting initpars into tuple
        self.initpars = (U,V,slice3d_cs_weighted,slice3d_shape,stage,rp)

    def solve(self,mu=1000,noiseless=True,max_iter=100):

        (U,V,slice3d_cs_weighted,slice3d_shape,s,rp) = self.initpars

        factors = self.decomp_factors
        hankel = np.matrix(U.dot(V.H))

        slice3d_orig_part = copy.deepcopy(slice3d_cs_weighted)
        # init lagrangian update
        lagr = np.matrix(np.zeros(hankel.shape,dtype=DTYPE))
        #lagr = copy.deepcopy(hankel)
        us = (U.H.dot(U)).shape
        vs = (V.H.dot(V)).shape
        Iu = np.eye(us[0],us[1],dtype=DTYPE)
        Iv = np.eye(vs[0],vs[0],dtype=DTYPE)


        for _ in range(max_iter):

            # taking the averages from tha hankel structure and rebuild
            hankel_inferred_part = np.multiply(U.dot(V.H)-lagr,\
                                                self.hankel_mask_inv)  
            if rp['recontype'] in ['k-t','kx-ky','kx-ky_angio']:
                slice3d_inferred_part = decompose_hankel_2d(hankel_inferred_part,\
                                                    slice3d_shape,s,factors,rp)

                slice3d = slice3d_orig_part + slice3d_inferred_part
                hankel = compose_hankel_2d(slice3d,rp)
            elif rp['recontype'] == 'something else':
                raise(Exception('not implemented'))
            # updating U,V and the lagrangian
            U = mu*(hankel+lagr).dot(V).dot(np.linalg.inv(Iv+mu*V.H.dot(V)))
            V = mu*((hankel+lagr).H).dot(U).dot(np.linalg.inv(Iu+mu*U.H.dot(U)))
            lagr = hankel - U.dot(V.H) + lagr

            if self.realtimeplot == True:
                self.rtplot.update_data(np.absolute(U.dot(V.H)))

        return U.dot(V.H)
    
    def solve_CUDA(self,mu=1000,noiseless=True,max_iter=100):

        pass
