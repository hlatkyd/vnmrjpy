import unittest
import vnmrjpy  as vj
import imageio
import numpy as np

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
        filled = vj.aloha.lowranksolvers.svt(img_masked,realtimeplot=True)
        
