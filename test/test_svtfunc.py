import unittest
import vnmrjpy  as vj
import imageio
import numpy as np
import sys

# working, but convergence is hard...

class Test_SVT(unittest.TestCase):
    
    def test_svt_func(self):
        
        # read boat image
        image = '/boat.png'
        thresh = 0.8
        img = imageio.imread(vj.pics+image)
        mask = np.random.rand(img.shape[0],img.shape[1])
        mask[mask >= thresh] = 1
        mask[mask < thresh] = 0
        img_masked = np.multiply(img, mask)
        filled = vj.aloha.lowranksolvers.svt(img_masked,known_data=img_masked,\
                                            tau=50000,realtimeplot=True)
        #svt = vj.aloha.SingularValueThresholding(img_masked,realtimeplot=True,\
        #                                        tau=50000,max_iter=1000,delta=1)
        #svt.solve()
        
        
