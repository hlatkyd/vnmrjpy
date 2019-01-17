import copy
import numpy as np
import sys
import vnmrjpy as vj
import time

class Admm():
    """Alternating Direction Method of Multipliers solver for Aloha


    Tuned for ALOHA MRI reconstruction framework, not for general use yet.
    Lmafit estimates the rank, then Admm is used to enforce the hankel structure

    refs.: Aloha papers ?
            Admm paper ?
    """
    def __init__(self,U,V,fiber_stage_known,stage,rp,\
                mu=1000,\
                realtimeplot=False,\
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
        #TODO change the old hankel compose-decompose to the new one
        fiber_shape = rp['fiber_shape']
        #removed np matrix
        # a.H syntax is changed to a.conj().T
        #U = np.matrix(U)
        #V = np.matrix(V)
        # make ankel mask out of known elmenets
        hankel_mask = np.absolute(vj.aloha.construct_hankel(fiber_stage_known,rp))
        hankel_mask[hankel_mask != 0] = 1
        hankel_mask = np.array(hankel_mask,dtype='complex64')
        hankel_mask_inv = np.ones(hankel_mask.shape) - hankel_mask

        self.hankel_mask = hankel_mask
        self.hankel_mask_inv = hankel_mask_inv
        # real time plotting for debugging purposes
        self.realtimeplot = realtimeplot
        if realtimeplot == True:
            self.rtplot = vj.util.RealTimeImshow(np.absolute(U.dot(V.conj().T)))

        # putting initpars into tuple
        self.initpars = (U,V,fiber_stage_known,stage,rp,mu,noiseless)

    def solve(self,max_iter=100):
        """The actual Admm iteration.
        
        Returns:
            hankel = U.dot(V.H) (np.matrix)
        """
        (U,V,fiber_stage,s,rp,mu,noiseless) = self.initpars

        hankel = U.dot(V.conj().T)

        fiber_orig_part = copy.deepcopy(fiber_stage)
        # init lagrangian update
        lagr = np.zeros(hankel.shape,dtype='complex64')
        #lagr = copy.deepcopy(hankel)
        us = (U.conj().T.dot(U)).shape
        vs = (V.conj().T.dot(V)).shape
        Iu = np.eye(us[0],us[1],dtype='complex64')
        Iv = np.eye(vs[0],vs[0],dtype='complex64')


        for _ in range(max_iter):
        
            #start = time.time()
            # taking the averages from tha hankel structure and rebuild
            hankel_inferred_part = np.multiply(\
                                U.dot(V.conj().T)-lagr,self.hankel_mask_inv)  
            #dtime = time.time()
            fiber_inferred_part = vj.aloha.deconstruct_hankel(\
                                hankel_inferred_part,s,rp)
            #print('deconstruct time {}'.format(time.time()-dtime))
            fiber = fiber_orig_part + fiber_inferred_part
            hankel = vj.aloha.construct_hankel(fiber,rp)
            # updating U,V and the lagrangian
            #TODO consider multidot....
            U = mu*(hankel+lagr).dot(V).dot(\
                                np.linalg.inv(Iv+mu*V.conj().T.dot(V)))
            V = mu*((hankel+lagr).conj().T).dot(U).dot(\
                                np.linalg.inv(Iu+mu*U.conj().T.dot(U)))
            lagr = hankel - U.dot(V.conj().T) + lagr

            if self.realtimeplot == True:
                self.rtplot.update_data(np.absolute(U.dot(V.conj().T)))
            #print('admm iter time {}'.format(time.time()-start))

        return U.dot(V.conj().T)
    
