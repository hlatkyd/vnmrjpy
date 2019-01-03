import numpy as np
import vnmrjpy as vj
from numba import jit
import sys
import copy

def pyramidal_solve_kt(slice3d,\
                    slice3d_orig,\
                    slice3d_shape,\
                    weights_list,\
                    factors,\
                    rp):
    """Solves a k-t slice: dim0=receivers,dim1=kx,dim2=t
    """ 
    #init
    lmafit_tol_list = vj.config['lmafit_tol']
    solver = rp['solver']
    slice3d_cs = copy.deepcopy(slice3d)
    kspace_complete = copy.deepcopy(slice3d)
    kspace_complete_stage = copy.deepcopy(slice3d)

    for s in range(rp['stages']):
        # init from previous stage

        kspace_init, center = vj.aloha.kspace_pyramidal_init(\
                                            slice3d,s)
        kspace_init_zerofilled, center2 = vj.aloha.kspace_pyramidal_init(\
                                            slice3d_cs,s)
        #kspace_weighing     
        kspace_weighted = vj.aloha.apply_pyramidal_weights_kxt(\
                                            kspace_init,\
                                            weights_list[s],\
                                            rp)
        kspace_zerofilled_weighted = vj.aloha.apply_pyramidal_weights_kxt(\
                                            kspace_init,\
                                            weights_list[s],\
                                            rp)
        #hankel formation
        hankel = vj.aloha.compose_hankel_2d(kspace_weighted,rp)
        hankel_zerofilled = vj.aloha.compose_hankel_2d(\
                                kspace_zerofilled_weighted,rp)
        #low rank matrix completion
        if solver == 'svt':
            svtsolver = SVTSolver(hankel,\
                            tau=None,\
                            delta=None,\
                            epsilon=1e-4,\
                            max_iter=500)
            hankel = vj.aloha.svtsolver.solve()
        elif solver == 'lmafit':
            # initialize with LMaFit
            lmafit = vj.aloha.Lmafit(hankel,\
                            known_data=hankel_zerofilled,\
                            verbose=False,\
                            realtimeplot=False,\
                            tol=lmafit_tol_list[s])
            X,Y,obj = lmafit.solve(max_iter=500)
            admm = vj.aloha.Admm(X,Y.H,\
                        kspace_zerofilled_weighted,\
                        slice3d_shape,\
                        s,\
                        rp,\
                        realtimeplot=False)
            hankel = admm.solve()
            #hankel = U.dot(V.H)
        else:
            raise(Exception('wrong solver'))
        # rearrange original from completed hankel
        kspace_weighted = vj.aloha.decompose_hankel_2d(hankel,\
                            slice3d_shape,s,factors,rp)
        kspace_complete_stage = \
                    vj.aloha.remove_pyramidal_weights_kxt(\
                                        kspace_weighted,\
                                        center,\
                                        weights_list[s])
        kspace_complete = vj.aloha.finalize_pyramidal_stage_kt(\
                                kspace_complete_stage,\
                                kspace_complete,\
                                slice3d, s, rp)    
                           
    kspace_complete = vj.aloha.restore_center(kspace_complete, slice3d)

    return kspace_complete

def pyramidal_solve_kxky(slice3d,\
                    slice3d_orig,\
                    slice3d_shape,\
                    weights_list,\
                    factors,\
                    rp):
    """Solves a k-t slice: dim0=receivers,dim1=kx,dim2=t
    """ 
    #init
    lmafit_tol_list = vj.config['lmafit_tol']
    solver = rp['solver']
    slice3d_cs = copy.deepcopy(slice3d)
    kspace_complete = copy.deepcopy(slice3d)
    kspace_complete_stage = copy.deepcopy(slice3d)

    for s in range(rp['stages']):
        # init from previous stage

        kspace_init, center = vj.aloha.kspace_pyramidal_init(\
                                            slice3d,s)
        kspace_init_zerofilled, center2 = vj.aloha.kspace_pyramidal_init(\
                                            slice3d_cs,s)
        #kspace_weighing     
        kspace_weighted = vj.aloha.apply_pyramidal_weights_kxky(\
                                            kspace_init,\
                                            weights_list[s],\
                                            rp)
        kspace_zerofilled_weighted = vj.aloha.apply_pyramidal_weights_kxky(\
                                            kspace_init,\
                                            weights_list[s],\
                                            rp)
        #hankel formation
        hankel = vj.aloha.compose_hankel_2d(kspace_weighted,rp)
        hankel_zerofilled = vj.aloha.compose_hankel_2d(\
                                kspace_zerofilled_weighted,rp)
        #low rank matrix completion
        if solver == 'svt':
            svtsolver = SVTSolver(hankel,\
                            tau=None,\
                            delta=None,\
                            epsilon=1e-4,\
                            max_iter=500)
            hankel = vj.aloha.svtsolver.solve()
        elif solver == 'lmafit':
            # initialize with LMaFit
            lmafit = vj.aloha.Lmafit(hankel,\
                            known_data=hankel_zerofilled,\
                            verbose=False,\
                            realtimeplot=False,\
                            tol=lmafit_tol_list[s])
            X,Y,obj = lmafit.solve(max_iter=500)
            admm = vj.aloha.Admm(X,Y.H,\
                        kspace_zerofilled_weighted,\
                        slice3d_shape,\
                        s,\
                        rp,\
                        realtimeplot=False)
            hankel = admm.solve()
            #hankel = U.dot(V.H)
        else:
            raise(Exception('wrong solver'))
        # rearrange original from completed hankel
        kspace_weighted = vj.aloha.decompose_hankel_2d(hankel,\
                            slice3d_shape,s,factors,rp)
        kspace_complete_stage = \
                    vj.aloha.remove_pyramidal_weights_kxky(\
                                        kspace_weighted,\
                                        center,\
                                        weights_list[s])
        kspace_complete = vj.aloha.finalize_pyramidal_stage(\
                                kspace_complete_stage,\
                                kspace_complete,\
                                slice3d, s, rp)    
                           
    kspace_complete = vj.aloha.restore_center(kspace_complete, slice3d)

    return kspace_complete
