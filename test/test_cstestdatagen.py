import vnmrjpy as vj
import unittest
import glob
import os

class Test_CsTestDataGenerator(unittest.TestCase):

    def test_testdatagen_gems(self):

        # directories
        fid_dir = sorted(glob.glob(vj.fids+'/gems_s*'))[0]
        base_dir = os.path.basename(fid_dir)[:-4]
        red = 4
        out_dir = vj.cs+'/'+base_dir+'_red'+str(red)+'.cs'
        # filepaths
        fid = fid_dir+'/fid'
        procpar = fid_dir+'/procpar'
        # generation
        gen = vj.util.CsTestDataGenerator(fid,procpar,reduction=red)
        gen.generate(savedir=out_dir)

    def test_testdatagen_angio(self):

        # directories
        fid_dir = sorted(glob.glob(vj.fids+'/ge3d_angio*'))[0]
        base_dir = os.path.basename(fid_dir)[:-4]
        #cs reduction
        red = 8
        out_dir = vj.cs+'/'+base_dir+'_red'+str(red)+'.cs'
        # filepaths
        fid = fid_dir+'/fid'
        procpar = fid_dir+'/procpar'
        # generation
        gen = vj.util.CsTestDataGenerator(fid,procpar,reduction=red)
        gen.generate(savedir=out_dir)

    def test_testdatagen_ge3d(self):

        # directories
        fid_dir = sorted(glob.glob(vj.fids+'/ge3d_s*'))[0]
        base_dir = os.path.basename(fid_dir)[:-4]
        red = 4
        out_dir = vj.cs+'/'+base_dir+'_red'+str(red)+'.cs'
        # filepaths
        fid = fid_dir+'/fid'
        procpar = fid_dir+'/procpar'
        # generation
        gen = vj.util.CsTestDataGenerator(fid,procpar,reduction=red)
        gen.generate(savedir=out_dir)

    def test_testdatagen_mems(self):

        # directories
        fid_dir = sorted(glob.glob(vj.fids+'/mems*'))[0]
        base_dir = os.path.basename(fid_dir)[:-4]
        red = 6
        out_dir = vj.cs+'/'+base_dir+'_red'+str(red)+'.cs'
        # filepaths
        fid = fid_dir+'/fid'
        procpar = fid_dir+'/procpar'
        # generation
        gen = vj.util.CsTestDataGenerator(fid,procpar,reduction=red)
        gen.generate(savedir=out_dir)
