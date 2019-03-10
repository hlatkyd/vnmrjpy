import unittest
import vnmrjpy as vj
import nibabel as nib
import numpy as np
import glob

class Test_nifti(unittest.TestCase):

    def test_gems(self):

        vj.config['verbose'] = False
        seq = glob.glob(vj.config['fids_dir']+'/gems*axial_*')[0]
        varr = vj.core.read_fid(seq)
        
        varr.to_kspace().to_imagespace()
        #varr = vj.core.niftitools._set_nifti_header(varr)
        niftiout = vj.config['testresults_dir']+'/new'
        niftiout = niftiout + '/testnifti'
        vj.core.write.write_nifti(varr,niftiout) 


        
