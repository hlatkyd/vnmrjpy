import unittest
import numpy as np
import vnmrjpy as vj
import glob
import os

class Test_niftiwriter(unittest.TestCase):

    def test_niftiwriter_ge3d(self):

        # easy 2d io
        ge3d_list = glob.glob(vj.config['dataset_dir']+'/debug/orient/ge3d*')
        ge3d = [i for i in ge3d_list if 'cor90' in i][0]
        procpar = ge3d+'/procpar'
        rdr = vj.io.FidReader(ge3d)
        image, tempaffine = rdr.make_image()
        outdir = vj.config['testresults_dir']+'/niftiwriter'
        outname = 'ge3d'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullout = outdir+'/'+outname
        writer = vj.io.NiftiWriter(image,procpar,\
                    input_space='scanner',output_space='rat_anatomical')
        writer.write(fullout)

    def test_niftiwriter_gems(self):

        # easy 2d io
        gems_list = glob.glob(vj.config['dataset_dir']+'/debug/orient/gems*')
        gems = [i for i in gems_list if 'axial90' in i][0]
        procpar = gems+'/procpar'
        rdr = vj.io.FidReader(gems)
        image, tempaffine = rdr.make_image()
        outdir = vj.config['testresults_dir']+'/niftiwriter'
        outname = 'gems'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullout = outdir+'/'+outname
        writer = vj.io.NiftiWriter(image,procpar,\
                    input_space='scanner',output_space='rat_anatomical')
        writer.write(fullout)

    def test_niftiwriter_gems_usual(self):

        # easy 2d io
        gems_list = glob.glob(vj.config['dataset_dir']+'/debug/orient/gems*')
        gems = [i for i in gems_list if 'usual' in i][0]
        procpar = gems+'/procpar'
        rdr = vj.io.FidReader(gems)
        image, tempaffine = rdr.make_image()
        outdir = vj.config['testresults_dir']+'/niftiwriter'
        outname = 'gems_usual'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fullout = outdir+'/'+outname
        writer = vj.io.NiftiWriter(image,procpar,\
                    input_space='scanner',output_space='rat_anatomical')
        writer.write(fullout)


    def test_7dnifti(self):

        mems_dir = glob.glob(vj.config['fids_dir']+'/mems*')[0]
        procpar = mems_dir+'/procpar'
        fid = mems_dir+'/fid'
        fid_data, fid_header = vj.io.FidReader(fid,procpar).read()
        kspace = vj.recon.KspaceMaker(fid_data,fid_header,procpar).make()
        imgspace = vj.recon.ImageSpaceMaker(kspace, procpar).make()
        img6d = vj.util.make_6d(imgspace,output_type='real-imag')
        writer = vj.io.NiftiWriter(img6d,procpar,input_space='local',\
                                    output_space='local')
        out6d = vj.config['testresults_dir']+'/niftiwriter/nifti6d'
        writer.write(out6d)
