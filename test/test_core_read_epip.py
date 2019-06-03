import vnmrjpy as vj
import numpy as np
import unittest
import glob

class Test_epip_read(unittest.TestCase):

    def test_epip_from_fid(self):

        seq = glob.glob(vj.config['fids_dir']+'/epip*1shot*')[0]
        print(seq)
        varr = vj.core.read_fid(seq)
        varr.to_kspace()
    def test_epip2shot_from_fid(self):

        seq = glob.glob(vj.config['fids_dir']+'/epip*2shot*')[0]
        varr = vj.core.read_fid(seq)
        varr.to_kspace()
