import vnmrjpy as vj
import unittest
import numpy as np
import nibabel as nib
import glob
import copy

# test on small number of slices only for 'speed'
SLC = 10
SLCNUM = 1

class Test_Aloha(unittest.TestCase):

    def _save_test_results(self,procpar,\
                                affine,\
                                kspace_orig,\
                                kspace_cs,\
                                kspace_filled):

        # TODO :  do this with imake maybe

        recon= vj.recon.ImageSpaceMaker(kspace, procpar)
        img = recon.recon()
        # TODO saveresults function

        pass

    def _load_data(self,data=None):

        if data == 'mems':
            seqdir = '/mems_s_2018111301_axial_0_0_0_01.cs'
            testdir = vj.cs+seqdir
            procpar = testdir + '/procpar'
            resultdir = vj.testresults+'/aloha/'+seqdir[:-3]+'.nifti'
        elif data == None:
            raise(Exception('Testdata not specified'))

        imag = []
        real = []
        imag_orig = []
        real_orig = []
    
        mask_img = nib.load(testdir+'/kspace_mask.nii.gz')
        affine = mask_img.affine
        mask = mask_img.get_fdata()

        for item in sorted(glob.glob(testdir+'/kspace_imag*')):
            data = nib.load(item).get_fdata()
            imag_orig.append(data)
            imag.append(np.multiply(data,mask))
        for item in sorted(glob.glob(testdir+'/kspace_real*')):
            data = nib.load(item).get_fdata()
            real_orig.append(data)
            real.append(np.multiply(data,mask))
        #for item in [imag,real,imag_orig,real_orig]:
        #    item = np.asarray(item)
        imag = np.asarray(imag)
        real = np.asarray(real)
        imag_orig = np.asarray(imag_orig)
        real_orig = np.asarray(real_orig)
        kspace_cs = np.vectorize(complex)(real,imag)
        kspace_orig = np.vectorize(complex)(real_orig,imag_orig)
    
        return (kspace_orig[...,SLC:SLC+SLCNUM,:],\
                kspace_cs[...,SLC:SLC+SLCNUM,:],\
                affine,\
                procpar)

    def test_mems(self):
    
        kspace_orig, kspace_cs, affine, procpar = self._load_data(data='mems')

        aloha = vj.aloha.Aloha(kspace_cs,procpar)
        print(aloha.rp)
        kspace_filled = aloha.recon() 
        self._save_test_results(procpar,affine,\
                                kspace_orig,kspace_cs,kspace_filled)

        
