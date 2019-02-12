import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt


class Test_Admm(unittest.TestCase):

    def test_on_mems_data(self):

        alohatest = vj.util.AlohaTest('offcenter',1)
        fiber_orig, fiber_cs = alohatest.load_test_cs_fiber('mems',recontype='k-t')
        rp = {'recontpye':'k-t','filter_size':(11,5),'vcboost':False,\
                'stage':3,'rcvrs':4,'fiber_shape':fiber_cs.shape}
        hankel_cs = vj.aloha.construct_hankel_2d(fiber_cs, rp)
        hankel_orig = vj.aloha.construct_hankel_2d(fiber_orig,rp)
        print('starting lmafit')
        X,Y,out = vj.aloha.lowranksolvers.lmafit(hankel_cs, \
                                            known_data=hankel_cs,\
                                            realtimeplot=False)
        stage=0
        print('starting admm')
        hankel_finished = vj.aloha.lowranksolvers.admm(X,Y.conj().T,\
                                                    fiber_cs,stage,rp,\
                                                    realtimeplot=True,\
                                                    max_iter=200,\
                                                    mu=1000,\
                                                    verbose=True)
        
    """
    def test_run_admm(self):
        # this is just to check if it runs...
        shape = (4,192,192)
        rp = {'recontype':'kx-ky','filter_size':(17,17),'vcboost':False,\
                'stage':3,'rcvrs':4,'fiber_shape':shape}
        # TODO load actual data
        stage = 0
        d1 = np.random.rand(*shape)
        d2 = np.random.rand(*shape)
        fiber_orig  = np.vectorize(complex)(d1,d2)
        mask = np.ones(fiber_orig.shape)
        mask[fiber_orig < 0.7] = 0
        fiber_known = np.multiply(fiber_orig,mask)
        hankel_known = vj.aloha.construct_hankel(fiber_known,rp)
        X,Y,out = vj.aloha.lowranksolvers.lmafit(hankel_known, \
                                            known_data=hankel_known,\
                                            realtimeplot=False)
        hankel_finished = vj.aloha.lowranksolvers.admm(X,Y.conj().T,\
                                                    fiber_orig,stage,rp,\
                                                    realtimeplot=True)
    """
    

    
