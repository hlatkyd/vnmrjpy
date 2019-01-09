import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt

RP={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False}
PLOTTING = False

class Test_hankelutils(unittest.TestCase):

    def test_construct_hankel(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False}
        indata = np.random.rand(4,128,21)
        hankel = vj.aloha.construct_hankel(indata,rp)
        
        self.assertEqual(hankel.shape,(1770,308))

    def test_deconstruct_hankel(self):
        
        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,128,21)}
        hankel = np.random.rand(1770,308)
        nd_data = vj.aloha.deconstruct_hankel(hankel, rp)
        self.assertEqual(nd_data.shape,(4,128,21))

    def test_make_kspace_weights(self):
        
        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,128,21),'stages':3}
        weights = vj.aloha.make_kspace_weights(rp)
        self.assertEqual(weights[1].shape,(4,64,21))

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':True,\
            'recontype':'kx-ky','fiber_shape':(8,128,128),'stages':3}
        weights = vj.aloha.make_kspace_weights(rp)
        self.assertEqual(weights[3].shape,(8,64,64))
        if PLOTTING == True:
            plt.imshow(np.absolute(weights[0][0,...]))
            plt.show()

    def test_init_kspace_stage(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'kx-ky','fiber_shape':(4,128,128),'stages':3}
        kspace = np.random.rand(4,128,128)
        stage = 1
        kspace_init = vj.aloha.init_kspace_stage(kspace,stage,rp)
        self.assertEqual(kspace_init.shape,(4,64,64))
    
