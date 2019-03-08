import vnmrjpy as vj
import numpy as np
import unittest
import glob
import matplotlib.pyplot as plt

class Test_core_to_imagespace(unittest.TestCase):

    def test_gems(self):

        seq = glob.glob(vj.config['fids_dir']+'/gems*axial_*')[0]
        varr = vj.core.read_fid(seq)
        varr.to_kspace().to_imagespace()
        #plt.imshow(np.absolute(varr.data[:,:,10,0,1]))
        #plt.show()
        self.assertEqual(varr.vdtype,'imagespace')

    def test_ge3d(self):

        seq = glob.glob(vj.config['fids_dir']+'/ge3d*axial_*')[0]
        varr = vj.core.read_fid(seq)
        varr.to_kspace().to_imagespace()
        self.assertEqual(varr.vdtype,'imagespace')
