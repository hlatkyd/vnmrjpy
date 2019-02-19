import os
import unittest
import vnmrjpy as vj
import glob

class Test_fid2nii(unittest.TestCase):

    def test_fid2nii(self):

        outdir = vj.config['testresults_dir']+'/bin'
        fid = glob.glob(vj.fids+'/gems*axial_0_0_0*')[0]
        out = outdir+'/'+fid.rsplit('/')[-1][:-4]
        os.system('fid2nii '+fid+' '+out)
