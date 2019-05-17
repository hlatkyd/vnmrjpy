import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from vnmrjpy.core import make_params3d, fit3d
from vnmrjpy.core import get_chisqr3d, get_best_values3d, get_best_fit3d

def _make_linear_data():
    pass
class Test_fit(unittest.TestCase):
    
    def test_minimize3d(self):
        
        def _linear_model(x,a,b):
            return a * x + b

        print('Testing vj.core.minimize3D ...')
        # set up data
        shape = (6,6,6,10)
        testarr = np.ones(shape)
        noise = np.random.rand(*shape)
        mult = np.array([float(i) for i in range(1,11)])
        y3d = testarr * mult + noise
        x3d = testarr * (mult-1)
        
        # set up lmfit.Parameters
        model = lmfit.Model(_linear_model)
        keys = model.param_names
        vals = 0, 1
        params_dict = dict(zip(keys, vals))
        params3d = make_params3d(model, shape[0:3], **params_dict)
        print(params3d[0,0,0])
        res3d = fit3d(_linear_model, y3d, params3d, x3d)

        """
        #p[lotting results
        res_choosen = res3d[5,5,5]
        x = x3d[5,5,5,:]
        y = y3d[5,5,5,:]
        plt.plot(x, y, 'bo')
        plt.plot(x,res_choosen.init_fit, 'k--')    
        plt.plot(x,res_choosen.best_fit, 'r--')    
        plt.show()
        """
        #print('best_values a: {}'.format(res3d[5,5,5].best_values['a']))
        #params = get_attribute3d(res3d, model)
        #print(type(list(res3d[0,0,0].best_values.values())[0]))
        a, b = get_best_values3d(res3d, model)
        c = get_chisqr3d(res3d)
        arr_fit = get_best_fit3d(res3d, shape[3])
        print(arr_fit.shape)
        #plt.imshow(arr_fit[2,2,:,:])
        #plt.show()
