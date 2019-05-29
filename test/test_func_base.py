import vnmrjpy as vj
import unittest
import numpy as np
import glob
import matplotlib.pyplot as plt
from vnmrjpy.func import concatenate, mask, average
import copy

class Test_base_mask(unittest.TestCase):

    def test_mask(self):

        seq = glob.glob(vj.config['fids_dir']+'/gems*axial_*')[0]
        varr = vj.core.read_fid(seq).to_kspace().to_imagespace()
        varr.to_anatomical()
        magn = vj.core.recon.ssos(varr.data)
        masked_varr = vj.func.base.mask(varr)

    def test_concatenate(self):

        seq = glob.glob(vj.config['fids_dir']+'/gems*axial_*')[0]
        varr = vj.core.read_fid(seq).to_kspace().to_imagespace()
        varr.to_anatomical()
        old = copy.copy(varr)
        concat = concatenate([varr,varr,varr])
        #print('previous shape: {}, now: {}'.format(old.data.shape, concat.data.shape))
        self.assertEqual(concat.data.shape[3],3)

    def test_average(self):

        seq = glob.glob(vj.config['fids_dir']+'/gems*axial_*')[0]
        varr = vj.core.read_fid(seq).to_kspace().to_imagespace()
        varr.to_anatomical()
        old = copy.copy(varr)
        avg = average([varr,varr,varr])
        #print('previous shape: {}, now: {}'.format(old.data.shape, avg.data.shape))
        self.assertEqual(avg.data.shape[3],1)


