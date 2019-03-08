import vnmrjpy as vj
import unittest
import numpy as np
import glob
import matplotlib.pyplot as plt

class Test_varray_to_kspace(unittest.TestCase):

    # test form im2D

    def test_gems_to_kspace(self):

        vj.config['verbose'] = False
        gems = glob.glob(vj.config['fids_dir']+'/gems*axial_*')[0]
        varr = vj.read_fid(gems)
        varr.to_kspace()
        self.assertEqual(varr.data.shape,(128,128,20,1,4))
        self.assertEqual(varr.space,'local')
        self.assertEqual(varr.vdtype,'kspace')

    def test_mgems_to_kspace(self):

        vj.config['verbose'] = False
        seq = glob.glob(vj.config['fids_dir']+'/mgems*axial_*')[0]
        varr = vj.read_fid(seq)
        varr.to_kspace()
        self.assertEqual(varr.data.shape,(128,128,20,8,4))
        self.assertEqual(varr.space,'local')
        self.assertEqual(varr.vdtype,'kspace')

    # test for im2Dfse

    def test_fsems_to_kspace(self):

        vj.config['verbose'] = False
        seq = glob.glob(vj.config['fids_dir']+'/fsems*axial_*')[0]
        varr = vj.read_fid(seq)
        varr.to_kspace()
        self.assertEqual(varr.data.shape,(128,128,20,1,4))
        self.assertEqual(varr.space,'local')
        self.assertEqual(varr.vdtype,'kspace')

    # test for im3D

    def test_ge3d_to_kspace(self):

        vj.config['verbose'] = False
        seq = glob.glob(vj.config['fids_dir']+'/ge3d*axial_*')[0]
        varr = vj.read_fid(seq)
        varr.to_kspace()
        self.assertEqual(varr.data.shape,(128,128,128,1,4))
        self.assertEqual(varr.space,'local')
        self.assertEqual(varr.vdtype,'kspace')
