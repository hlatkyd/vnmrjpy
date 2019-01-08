import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt

RP={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False}

class Test_hankelutils(unittest.TestCase):

    def test_construct_hankel(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False}
        indata = np.random.rand(4,128,21)
        hankel = vj.aloha.construct_hankel(indata,rp)
        
        self.assertEqual(hankel.shape,(1770,308))

    def test_deconstruc_hankel(self):
        
        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','orig_shape':(4,128,21)}
        hankel = np.random.rand(1770,308)
        vj.aloha.deconstruct_hankel(hankel, rp)
