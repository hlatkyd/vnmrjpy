import vnmrjpy as vj
import numpy as np
import unittest
import glob

class Test_Xrecon(unittest.TestCase):

    def test_xrecon(self):

        fid = glob.glob(vj.config['fids_dir']+'/gems*')[0]
        print(fid)
        varr = vj.read_fid(fid,load_data=False)
