import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt


class Test_Admm(unittest.TestCase):

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
                                            realtimeplot=True)
        hankel_finished = vj.aloha.lowranksolvers.admm(X,Y.conj().T,\
                                                    fiber_orig,stage,rp,\
                                                    realtimeplot=True)




