import copy
import numpy as np
import sys
import vnmrjpy as vj

class Admm():
    """Alternating Direction Method of Multipliers solver for Aloha


    Tuned for ALOHA MRI reconstruction framework, not for general use yet.
    Lmafit estimates the rank, then Admm is used to enforce the hankel structure

    refs.: Aloha papers ?
            Admm paper ?
    """
    def __init__(self,U,V,slice3d_cs_weighted,slice3d_shape,stage,rp,\
                mu=1000,\
                realtimeplot=True,\
                noiseless=True,\
                device='CPU'):
        """Initialize solver
        
        Args:
            U, V  (np.matrix) U*V.H is the estimated hankel matrix from Lmafit
            slice3d_cs_weighted : 
            slice3d_shape : 
            stage (int) : pyramidal decomposition stage ?? WHY??
            rp {dictionary} : Aloha recon parameters 
            realitmeplot (Boolean) option to plot each iteration

        """
        U = np.matrix(U)
        V = np.matrix(V)
        if rp['recontype'] in ['k-t','kx-ky','kx-ky_angio']:
            self.hankel_mask = np.absolute(compose_hankel_2d(slice3d_cs_weighted,rp))

            self.decomp_factors = vj.aloha.make_hankel_decompose_factors(\
                                    slice3d_shape,rp)
        elif rp['reontype'] == 'higher dim':
            raise(Exception('not implemented'))

        self.hankel_mask[self.hankel_mask != 0] = 1
        self.hankel_mask = np.array(self.hankel_mask,dtype='complex64')
        self.hankel_mask_inv = np.ones(self.hankel_mask.shape) - self.hankel_mask
        # real time plotting for debugging purposes
        self.realtimeplot = realtimeplot
        if realtimeplot == True:
            self.rtplot = vj.util.RealTimeImshow(np.absolute(U.dot(V.H)))

        # putting initpars into tuple
        self.initpars = (U,V,slice3d_cs_weighted,slice3d_shape,stage,rp,\
                        mu,noiseless)

    def solve(max_iter=100):
        """The actual Admm iteration.
        
        Returns:
            hankel = U.dot(V.H) (np.matrix)
        """
        (U,V,slice3d_cs_weighted,slice3d_shape,s,rp,mu,noiseless) = self.initpars

        factors = self.decomp_factors
        hankel = np.matrix(U.dot(V.H))

        slice3d_orig_part = copy.deepcopy(slice3d_cs_weighted)
        # init lagrangian update
        lagr = np.matrix(np.zeros(hankel.shape,dtype='complex64'))
        #lagr = copy.deepcopy(hankel)
        us = (U.H.dot(U)).shape
        vs = (V.H.dot(V)).shape
        Iu = np.eye(us[0],us[1],dtype='complex64')
        Iv = np.eye(vs[0],vs[0],dtype='complex64')


        for _ in range(max_iter):

            # taking the averages from tha hankel structure and rebuild
            hankel_inferred_part = np.multiply(U.dot(V.H)-lagr,\
                                                self.hankel_mask_inv)  
            if rp['recontype'] in ['k-t','kx-ky','kx-ky_angio']:
                slice3d_inferred_part = vj.aloha.decompose_hankel_2d(\
                            hankel_inferred_part,slice3d_shape,s,factors,rp)

                slice3d = slice3d_orig_part + slice3d_inferred_part
                hankel = vj.aloha.compose_hankel_2d(slice3d,rp)
            elif rp['recontype'] == 'something else':
                raise(Exception('not implemented'))
            # updating U,V and the lagrangian
            U = mu*(hankel+lagr).dot(V).dot(np.linalg.inv(Iv+mu*V.H.dot(V)))
            V = mu*((hankel+lagr).H).dot(U).dot(np.linalg.inv(Iu+mu*U.H.dot(U)))
            lagr = hankel - U.dot(V.H) + lagr

            if self.realtimeplot == True:
                self.rtplot.update_data(np.absolute(U.dot(V.H)))

        return U.dot(V.H)
    
