import vnmrjpy as vj
import unittest
import glob
import os

class Test_CsTestDataGenerator(unittest.TestCase):

    def test_testdatagen(self):

        # directories
        fid_dir = sorted(glob.glob(vj.fids+'/gems_s*'))[0]
        base_dir = os.path.basename(fid_dir)[:-4]
        out_dir = vj.cs+'/'+base_dir+'.cs'
        # filepaths
        fid = fid_dir+'/fid'
        procpar = fid_dir+'/procpar'
        # generation
        gen = vj.util.CsTestDataGenerator(fid,procpar)
        gen.generate(savedir=out_dir)
        
        print(base_dir)
        print(out_dir)
