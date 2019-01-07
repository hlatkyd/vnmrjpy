import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt

RP={'rcvrs':4,'filter_size':(11,13),'virtualcoilboost':False}

class Test_hankelutils(unittest.TestCase):

    def test_construct_hankel(self):

        indata = np.random.rand(4,128,21)
        hankel = vj.aloha.construct_hankel(indata,RP)
        
        #plt.imshow(np.absolute(hankel))
        #plt.show()
