import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters

class Test_fit(unittest.TestCase):
    
    def test_minimize3d(self):
        
        def _model(x,params):
            a = params[0]
            b = params[1]

        print('Testing vj.core.minimize3D ...')
        shape = (100,100,100,10)
        testarr = np.ones(shape)
        noise = np.random.rand(*shape)
        mult = np.array([float(i) for i in range(1,11)])
        testarr = testarr * mult + noise
        # model
        
        # creating residual
    

        pass


