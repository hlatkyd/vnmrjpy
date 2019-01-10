import unittest
import vnmrjpy as vj
import glob
import nibabel as nib

class Test_SkipintGenerator(unittest.TestCase):

    def test_generate_gems(self):

        reduction = 4
        gemsdir = sorted(glob.glob(vj.fids+'/gems*.fid'))[0]
        procpar = gemsdir+'/procpar'
        gen = vj.util.SkipintGenerator(procpar=procpar) 
        kmask = gen.generate_kspace_mask()
        self.assertEqual(len(kmask.shape),4)
        #nib.viewers.OrthoSlicer3D(kmask).show()

    def test_generate_ge3d(self):

        reduction = 4
        gemsdir = sorted(glob.glob(vj.fids+'/ge3d_s*.fid'))[0]
        procpar = gemsdir+'/procpar'
        gen = vj.util.SkipintGenerator(procpar=procpar) 
        kmask = gen.generate_kspace_mask()
        self.assertEqual(len(kmask.shape),4)
    
        #nib.viewers.OrthoSlicer3D(kmask).show()

    def test_generate_mge3d(self):

        reduction = 4
        gemsdir = sorted(glob.glob(vj.fids+'/mge3d*.fid'))[0]
        procpar = gemsdir+'/procpar'
        gen = vj.util.SkipintGenerator(procpar=procpar) 
        kmask = gen.generate_kspace_mask()
        self.assertEqual(len(kmask.shape),4)
    
        nib.viewers.OrthoSlicer3D(kmask).show()

    def test_skiptab_ge3d(self):
        """ Generate skiptab"""
        pass
