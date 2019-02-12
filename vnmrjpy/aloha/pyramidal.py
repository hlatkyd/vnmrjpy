import numpy as np
import vnmrjpy as vj
import sys
import copy
import matplotlib.pyplot as plt
import time

def pyramidal_k(kspace_fiber,weights,rp,\
                    realtimeplot=False):
    """ Pyramidal decomposition composit function for single k sparsity case.

    Main Aloha unit.
    One pyramidal decomposition and matrix completion is used at each readout
    point. Thus, the different calls for this function are independent and
    could be passed to the grid or GPU or whatever

    Args:
        kspace_fiber
        weights
        rp
    Return:
        kspace_fiber_complete
    """
    kspace_fiber_complete = copy.deepcopy(kspace_fiber)
    if rp['stages'] == 3:
        lmafit_tolerance = vj.config['lmafit_tol']
    else:
        raise(Exception('no lmafit tolerance at different number of stages\
                        than 3. please specify in config or pyramidal'))
    for s in range(rp['stages']):

        weight = weights[s]
        # init known data for the solvers
        fiber_known = vj.aloha.init_kspace_stage(kspace_fiber,s,rp)
        # saving known center
        fiber_known = vj.aloha.apply_kspace_weights(fiber_known,weight)
        hankel_known = vj.aloha.construct_lvl1_hankel(fiber_known,rp['filter_size'])
        # init data to be completed
        kspace_stage = vj.aloha.init_kspace_stage(kspace_fiber_complete,s,rp)
        kspace_stage = vj.aloha.apply_kspace_weights(kspace_stage,weight)
        hankel = vj.aloha.construct_lvl1_hankel(kspace_stage,rp['filter_size'])
       
        if rp['solver'] == 'svt':
            raise(Exception('not implemented'))
        elif rp['solver'] == 'lmafit':
            lmafit = vj.aloha.Lmafit(hankel,\
                                    known_data=hankel_known,\
                                    verbose=False,\
                                    realtimeplot=realtimeplot,\
                                    tol=lmafit_tolerance[s])
            X,Y,obj = lmafit.solve(max_iter=200)
            hankel = vj.aloha.lowranksolvers.admm(X,\
                                                Y.conj().T,\
                                                fiber_known,\
                                                s,\
                                                rp,\
                                                realtimeplot=realtimeplot,\
                                                max_iter=40)
        fiber = vj.aloha.deconstruct_hankel(hankel,s,rp)
        fiber = vj.aloha.remove_kspace_weights(fiber,weight)
        # replacing center
        kspace_fiber_complete = vj.aloha.finish_kspace_stage(\
                                        fiber,kspace_fiber_complete,0,rp)

    return kspace_fiber_complete

def pyramidal_kxky(kspace_fiber,weights,rp,\
                    realtimeplot=False):
    """ Pyramidal decomposition composit function for kx-ky sparsity case.

    Main Aloha unit.
    One pyramidal decomposition and matrix completion is used at each readout
    point. Thus, the different calls for this function are independent and
    could be passed to the grid or GPU or whatever

    Args:
        kspace_fiber
        weights
        rp
    Return:
        kspace_fiber_complete
    """
    kspace_fiber_complete = copy.deepcopy(kspace_fiber)
    if rp['stages'] == 3:
        lmafit_tolerance = vj.config['lmafit_tol']
    #TODO ugly hack correct this....
    elif rp['stages'] == 1:
        lmafit_tolerance = [vj.config['lmafit_tol'][2]]
    else:
        raise(Exception('no lmafit tolerance at different number of stages\
                        than 3. please specify in config or pyramidal'))
    for s in range(rp['stages']):

        for i in range(2):

            weight = weights[2*s+i]
            # init known data for the solvers
            fiber_known = vj.aloha.init_kspace_stage(kspace_fiber,s,rp)
            # saving known center
            fiber_known = vj.aloha.apply_kspace_weights(fiber_known,weight)
            hankel_known = vj.aloha.construct_hankel(fiber_known,rp)
            # init data to be completed
            kspace_stage = vj.aloha.init_kspace_stage(kspace_fiber_complete,s,rp)
            kspace_stage = vj.aloha.apply_kspace_weights(kspace_stage,weight)
            hankel = vj.aloha.construct_hankel(kspace_stage,rp)
           
            if rp['solver'] == 'svt':
                raise(Exception('not implemented'))
            elif rp['solver'] == 'lmafit':
                lmafit = vj.aloha.Lmafit(hankel,\
                                        known_data=hankel_known,\
                                        verbose=False,\
                                        realtimeplot=realtimeplot,\
                                        tol=lmafit_tolerance[s])
                X,Y,obj = lmafit.solve(max_iter=500)
                print('lmafit ready')
                hankel = vj.aloha.lowranksolvers.admm(X,\
                                                    Y.conj().T,\
                                                    fiber_known,\
                                                    s,\
                                                    rp,\
                                                    realtimeplot=realtimeplot)
            fiber = vj.aloha.deconstruct_hankel(hankel,s,rp)
            fiber = vj.aloha.remove_kspace_weights(fiber,weight)
            # this is for replacing zerofreq lines
            if i == 0:
                weight_x = copy.copy(weight)
                fiber_wx = copy.copy(fiber)
            if i == 1:
                weight_y = weight
                fiber_wy = fiber

        # averaging kx and ky completed fibers
        fiber = vj.aloha.avg_xy_fibers(fiber_wx,fiber_wy,weight_x,weight_y)
        # replacing center
        kspace_fiber_complete = vj.aloha.finish_kspace_stage(\
                                        fiber,kspace_fiber_complete,i,rp)

    return kspace_fiber_complete
            

def pyramidal_kt(kspace_fiber,weights,rp,\
                    realtimeplot=False):
    """ Pyramidal decomposition composit function for k-t sparsity.

    Main Aloha unit.
    One pyramidal decomposition and matrix completion is used at each readout
    point. Thus, the different calls for this function are independent and
    could be passed to the grid or GPU or whatever

    Args:
        kspace_fiber
        weights
        rp
    Return:
        kspace_fiber_complete
    """
    kspace_fiber_complete = copy.deepcopy(kspace_fiber)
    if rp['stages'] == 3:
        lmafit_tolerance = vj.config['lmafit_tol']
    else:
        raise(Exception('no lmafit tolerance at different number of stages\
                        than 3. please specify in config or pyramidal'))
    for s in range(rp['stages']):

        weight = weights[s]
        # init known data for the solvers
        fiber_known = vj.aloha.init_kspace_stage(kspace_fiber,s,rp)
        fiber_known = vj.aloha.apply_kspace_weights(fiber_known,weight)
        hankel_known = vj.aloha.construct_hankel(fiber_known,rp)
        # init data to be completed
        kspace_stage = vj.aloha.init_kspace_stage(kspace_fiber_complete,s,rp)
        kspace_stage = vj.aloha.apply_kspace_weights(kspace_stage,weight)
        hankel = vj.aloha.construct_hankel(kspace_stage,rp)
       
        if rp['solver'] == 'svt':
            raise(Exception('not implemented'))
        elif rp['solver'] == 'lmafit':
            lmafit = vj.aloha.Lmafit(hankel,\
                                    known_data=hankel_known,\
                                    verbose=False,\
                                    realtimeplot=realtimeplot,\
                                    tol=lmafit_tolerance[s])
            X,Y,obj = lmafit.solve(max_iter=500)
            """ old
            admm = vj.aloha.Admm(X,Y.conj().T,fiber_known, s,rp,\
                                    realtimeplot=False)
            hankel = admm.solve()
            """
            hankel = vj.aloha.lowranksolvers.admm(X,\
                                                Y.conj().T,\
                                                fiber_known,\
                                                s,\
                                                rp,\
                                                realtimeplot=realtimeplot)

        fiber = vj.aloha.deconstruct_hankel(hankel,s,rp)
        fiber = vj.aloha.remove_kspace_weights(fiber,weight)
        kspace_fiber_complete = vj.aloha.finish_kspace_stage(\
                                    fiber,kspace_fiber_complete,0,rp)
        
    return kspace_fiber_complete
            

# deprecated-------------------------------------------------------------------
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
        weights_ind = 0
        # go through each stage twice
        # weighting is done in one direction at a time
        for _ in range(2):
            # init from previous stage

            kspace_init, center = vj.aloha.kspace_pyramidal_init(\
                                                slice3d,s)
            kspace_init_zerofilled, center2 = vj.aloha.kspace_pyramidal_init(\
                                                slice3d_cs,s)
            #kspace_weighing     
            kspace_weighted = vj.aloha.apply_pyramidal_weights_kxky(\
                                                kspace_init,\
                                                weights_list[s],\
                                                weights_ind,\
                                                rp)
            kspace_zerofilled_weighted = vj.aloha.apply_pyramidal_weights_kxky(\
                                                kspace_init,\
                                                weights_list[s],\
                                                weights_ind,\
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
                                            weights_list[s],\
                                            weights_ind)
            kspace_complete = vj.aloha.finalize_pyramidal_stage(\
                                    kspace_complete_stage,\
                                    kspace_complete,\
                                    slice3d, s, rp)    
            weights_ind = 1
                           
    kspace_complete = vj.aloha.restore_center(kspace_complete, slice3d)

    return kspace_complete
