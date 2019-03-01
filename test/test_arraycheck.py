import numpy as np
import unittest
import vnmrjpy as vj
import glob
import nibabel as nib

class Test_array_param(unittest.TestCase):

    def test_calc_array_length(self):

        fid_dir = glob.glob(vj.config['dataset_dir']+'/arrayed/*arrayed_flip1*')[0]
        procpar = fid_dir+'/procpar'
        fid = fid_dir+'/fid'
        p = vj.io.ProcparReader(procpar).read()
        te_num = len(p['flip1'])
        fid_data, fid_header = vj.io.FidReader(fid,procpar).read()
        l = vj.util.calc_array_length(fid_data.shape,procpar)
        self.assertEqual(l,te_num)

    def test_array_gems_kmake(self):

        fid_dir = glob.glob(vj.config['dataset_dir']+'/arrayed/*array_flip1*')[0]
        procpar = fid_dir+'/procpar'
        fid = fid_dir+'/fid'
        image, affine = vj.io.FidReader(fid,procpar).make_image()
        ni = nib.viewers.OrthoSlicer3D(image, affine=affine)
        ni.show()
        
    
    
