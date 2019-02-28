import unittest
import numpy as np
import vnmrjpy as vj
import glob
import os

class Test_niftiwriter(unittest.TestCase):

    def test_niftiwriter(self):

        # easy 2d io
        #gems = glob.glob(vj.config['fids_dir']+'/gems*axial90_0_90_0_01.fid')[0]
        #gems_list = glob.glob(vj.config['dataset_dir']+'/debug/orient/gems*')
        #gems = [i for i in gems_list if 'axial90' in i][0]
        ge3d_list = glob.glob(vj.config['dataset_dir']+'/debug/orient/ge3d*')
        ge3d = [i for i in ge3d_list if 'cor90' in i][0]
        print(ge3d)
        procpar = ge3d+'/procpar'
        rdr = vj.io.FidReader(ge3d)
        image, tempaffine = rdr.make_image()
        #image = vj.util.to_scanner_space(image, procpar)

        outdir = vj.config['testresults_dir']+'/niftiwriter'
        outname = 'ge3d_02'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullout = outdir+'/'+outname
        writer = vj.io.NiftiWriter(image,procpar)
        writer.write(fullout)
