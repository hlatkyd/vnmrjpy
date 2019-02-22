import vnmrjpy as vj
import numpy as np
import unittest
import glob
import os
import matplotlib.pyplot as plt


class Test_kmake_epip(unittest.TestCase):

    def test_kmake_from_fid(self):

        #epip_dir = sorted(glob.glob(vj.fids+'/epip_s_2018111301_1shot_lowres_axial_0*'))[0]
        epip_dir = sorted(glob.glob(vj.fids+'/epip_s_2018111301_2shot_axial_0*'))[0]
        fid = epip_dir+'/fid'
        procpar = epip_dir+'/procpar'
        reader = vj.io.FidReader(epip_dir)
        reader.print_header()
        fiddata, fidhdr = reader.read()
        print('epip fid data : {}'.format(fiddata.shape))
        
        kmaker = vj.recon.KspaceMaker(fiddata, fidhdr, procpar, verbose=True) 
        kspace = kmaker.make()
        
        out_dir = vj.config['testresults_dir']+'/kmake/epip'
        outfile_name_k = 'epip_test_kspace'
        #outfile_name_i = 'epip_test_imgspace'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        outpath_k = out_dir+'/'+outfile_name_k
        #outpath_i = out_dir+'/'+outfile_name_i

        #imaker = vj.recon.ImageSpaceMaker(kspace, procpar)
        #imgspace = imaker.make()
        writer = vj.io.NiftiWriter(np.absolute(kspace[1,...]),procpar)
        writer.write(outpath_k)
        #writer = vj.io.NiftiWriter(np.absolute(imgspace[1,...]),procpar)
        #writer.write(outpath_i)

