import unittest
import vnmrjpy as vj
import numpy as np

class Test_Pyramidal(unittest.TestCase):

    def test_pyramidal_kt(self):

        rp = {'rcvrs':4,'fiber_shape':(4,128,21),'recontype':'k-t',\
                'stages':3,'solver':'lmafit','virtualcoilboost':False,\
                'filter_size':(11,7)} 
        a = np.random.rand(4,128,21)
        b = np.random.rand(4,128,21)
        kspace_fiber = np.vectorize(complex)(a,b)
        weights = vj.aloha.make_kspace_weights(rp)
        kspace_fin = vj.aloha.pyramidal_kt(kspace_fiber,weights,rp)

        self.assertEqual(kspace_fin.shape,kspace_fiber.shape)

    def test_pyramidal_kxky(self):

        rp = {'rcvrs':4,'fiber_shape':(4,128,128),'recontype':'kx-ky',\
                'stages':3,'solver':'lmafit','virtualcoilboost':False,\
                'filter_size':(11,11)} 
        a = np.random.rand(4,128,128)
        b = np.random.rand(4,128,128)
        kspace_fiber = np.vectorize(complex)(a,b)
        weights = vj.aloha.make_kspace_weights(rp)
        kspace_fin = vj.aloha.pyramidal_kxky(kspace_fiber,weights,rp)

        self.assertEqual(kspace_fin.shape,kspace_fiber.shape)
