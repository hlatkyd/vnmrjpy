import vnmrjpy as vj
import unittest
import matplotlib.pyplot as plt
import numpy as np


class Test_AlohaTest_fiberload(unittest.TestCase):

    def test_load_test_cs_fiber(self):

        alohatest = vj.util.AlohaTest('offcenter',1)
        fiber_orig, fiber_cs = alohatest.load_test_cs_fiber('gems',recontype='k')
        
        imgmax = np.max(np.absolute(fiber_orig))

        plt.subplot(1,2,1)
        plt.imshow(np.absolute(fiber_orig[0,...]),cmap='gray', vmin=0,vmax=10)
        plt.title('original abs')
        plt.subplot(1,2,2)
        plt.imshow(np.absolute(fiber_cs[0,...]),cmap='gray', vmin=0,vmax=10)
        plt.title('cs abs')
        plt.show()

         
