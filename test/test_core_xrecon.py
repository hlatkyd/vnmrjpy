import vnmrjpy as vj
import numpy as np
import unittest
import glob

class Test_Xrecon(unittest.TestCase):

    def test_xrecon(self):

        fid = glob.glob(vj.config['dataset_dir']+'/xrecontest/ge3d*')[0]
        varr = vj.read_fid(fid,load_data=False)
        varr.to_kspace(method='xrecon')
        self.assertEqual(varr.pd['rawIM'],'y')
        print(varr.data.shape)
