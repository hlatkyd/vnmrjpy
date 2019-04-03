import vnmrjpy as vj
import numpy as np
import unittest
import glob
import matplotlib.pyplot as plt

class Test_Wasabi(unittest.TestCase):

    def test_wasabi(self):

        seq = glob.glob(vj.config['dataset_dir']+\
                        '/parameterfit/wasabi/gems*test05*.fid')[0]
        seq_norm = glob.glob(vj.config['dataset_dir']+\
                        '/parameterfit/wasabi/gems*norm02*fid')[0]

        varr = vj.core.read_fid(seq).to_kspace()
        varr.to_imagespace().to_anatomical()
        varr_norm = vj.core.read_fid(seq_norm).to_kspace()
        varr_norm.to_imagespace().to_anatomical()
        # reduce data along slice dim to make test faster
        mask = vj.core.recon.ssos(varr_norm.data)
        # naive mask for testing
        mask[mask < 5] = 0
        mask[mask != 0] = 1
        (res, err) = vj.func.WASABI(varr, varr_norm, mask=mask) 
        out_b0 = vj.config['testresults_dir']+'/wasabi/b0map'
        out_b1 = vj.config['testresults_dir']+'/wasabi/b1map'
        out_c = vj.config['testresults_dir']+'/wasabi/cmap'
        out_d = vj.config['testresults_dir']+'/wasabi/dmap'
        # errors
        out_b0_err = vj.config['testresults_dir']+'/wasabi/b0map_err'
        out_b1_err = vj.config['testresults_dir']+'/wasabi/b1map_err'
        out_c_err = vj.config['testresults_dir']+'/wasabi/cmap_err'
        out_d_err = vj.config['testresults_dir']+'/wasabi/dmap_err'

        vj.core.write_nifti(res[0],out_b0)
        vj.core.write_nifti(res[1],out_b1)
        vj.core.write_nifti(res[2],out_c)
        vj.core.write_nifti(res[3],out_d)

        vj.core.write_nifti(err[0],out_b0_err)
        vj.core.write_nifti(err[1],out_b1_err)
        vj.core.write_nifti(err[2],out_c_err)
        vj.core.write_nifti(err[3],out_d_err)
