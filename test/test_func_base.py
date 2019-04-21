import vnmrjpy as vj
import unittest
import numpy as np
import glob
import matplotlib.pyplot as plt

class Test_base_mask(unittest.TestCase):

    def test_mask(self):

        seq = glob.glob(vj.config['fids_dir']+'/gems*axial_*')[0]
        print(seq)
        varr = vj.core.read_fid(seq).to_kspace().to_imagespace()
        varr.to_anatomical()
        print(varr.data.shape)
        magn = vj.core.recon.ssos(varr.data)
        print(magn.shape)
        print(varr.sdims)
        print(varr.dims)
        mask = vj.func.base.mask(varr)
        print('Mask should be made with BET. Consider this again....')
