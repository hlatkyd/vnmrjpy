import numpy as np
from numba import jit
import sys

def pyramidal_solve_kt(slice3d,\
                    slice3d_orig,\
                    slice3d_shape,\
                    weights_list,\
                    factors,\
                    rp):
    """
    Solves a k-t slice: dim0=receivers,dim1=kx,dim2=t
    """ 
    #init
    lmafit_tol_list = [5/10**(s+2) for s in range(rp['stages'])]
    solver = rp['solver']
    slice3d_cs = copy.deepcopy(slice3d)
    kspace_complete = copy.deepcopy(slice3d)
    kspace_complete_stage = copy.deepcopy(slice3d)
    for s in range(rp['stages']):
        # init from previous stage

        kspace_init, center = kspace_pyramidal_init(\
                                            slice3d,s)
        kspace_init_zerofilled, center2 = kspace_pyramidal_init(\
                                            slice3d_cs,s)
        #kspace_weighing     
        kspace_weighted = apply_pyramidal_weights_kxt(\
                                            kspace_init,\
                                            weights_list[s],\
                                            rp)
        kspace_zerofilled_weighted = apply_pyramidal_weights_kxt(\
                                            kspace_init,\
                                            weights_list[s],\
                                            rp)
        #hankel formation
        hankel = compose_hankel_2d(kspace_weighted,rp)
        hankel_zerofilled = compose_hankel_2d(\
                                kspace_zerofilled_weighted,rp)
        #low rank matrix completion
        if solver == 'svt':
            svtsolver = SVTSolver(hankel,\
                            tau=None,\
                            delta=None,\
                            epsilon=1e-4,\
                            max_iter=500)
            hankel = svtsolver.solve()
        elif solver == 'lmafit':
            # initialize with LMaFit
            lmafit = LMaFit(hankel,hankel_zerofilled,\
                            verbose=False,\
                            realtimeplot=False,\
                            tol=lmafit_tol_list[s])
            X,Y,obj = lmafit.solve(max_iter=500)
            admm = ADMM(X,Y.H,\
                        kspace_zerofilled_weighted,\
                        slice3d_shape,\
                        s,\
                        rp,\
                        realtimeplot=False)
            hankel = admm.solve(max_iter=100)
            #hankel = U.dot(V.H)
        else:
            raise(Exception('wrong solver'))
        # rearrange original from completed hankel
        kspace_weighted = decompose_hankel_2d(hankel,\
                            slice3d_shape,s,factors,rp)
        kspace_complete_stage = \
                    remove_pyramidal_weights_kxt(\
                                        kspace_weighted,\
                                        center,\
                                        weights_list[s])
        kspace_complete = finalize_pyramidal_stage(\
                                kspace_complete_stage,\
                                kspace_complete,\
                                slice3d, s, rp)    
                           
    kspace_complete = restore_center(kspace_complete, slice3d)

    return kspace_complete
