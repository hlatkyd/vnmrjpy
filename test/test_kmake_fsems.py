import vnmrjpy as vj
import numpy as np
import unittest
import glob
import os
import matplotlib.pyplot as plt


class Test_kmake_fsems(unittest.TestCase):

    def test_kmake_from_fid(self):

        fsems_dir = sorted(glob.glob(vj.fids+'/fsems_s_2018111301_axial_0*'))[0]
        fid = fsems_dir+'/fid'
        procpar = fsems_dir+'/procpar'
        fiddata, fidhdr = vj.io.FidReader(fsems_dir).read()
        
        kmaker = vj.recon.KspaceMaker(fiddata, fidhdr, procpar, verbose=True) 
        kspace = kmaker.make()
        
        out_dir = vj.config['testresults_dir']+'/kmake/fsems'
        outfile_name_k = 'fsems_test_kspace'
        outfile_name_i = 'fsems_test_imgspace'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        outpath_k = out_dir+'/'+outfile_name_k
        outpath_i = out_dir+'/'+outfile_name_i

        imaker = vj.recon.ImageSpaceMaker(kspace, procpar)
        imgspace = imaker.make()
        writer = vj.io.NiftiWriter(np.absolute(kspace[1,...]),procpar)
        writer.write(outpath_k)
        writer = vj.io.NiftiWriter(np.absolute(imgspace[1,...]),procpar)
        writer.write(outpath_i)

