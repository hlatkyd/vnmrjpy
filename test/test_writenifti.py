import unittest
import numpy as np
import vnmrjpy as vj
import glob
import os

class Test_niftiwriter(unittest.TestCase):

    def test_niftiwriter(self):

        # easy 2d io
        gems = glob.glob(vj.config['fids_dir']+'/gems*axial90_0_90_0_01.fid')[0]
        procpar = gems+'/procpar'
        rdr = vj.io.FidReader(gems)
        image, tempaffine = rdr.make_image()
        #image = vj.util.to_scanner_space(image, procpar)

        outdir = vj.config['testresults_dir']+'/niftiwriter'
        outname = 'gems'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullout = outdir+'/'+outname
        writer = vj.io.NiftiWriter(image,procpar)
        writer.write(fullout)
