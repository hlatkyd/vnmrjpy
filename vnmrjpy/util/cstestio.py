import vnmrjpy as vj
import numpy as np
import glob
import nibabel as nib
import copy
"""
Collection of utility functions for systematic Aloha testing

"""

class AlohaTest():
    """This is for consistent data handling during the testphase
    """
    def __init__(self, slc, slcnum):

        # TODO
        self.slc = slc  # can be slice index, or 'center', 'offcenter'
        self.slcnum = slcnum

    def load_test_cs_fiber(self,datatype, recontype=None):
        """Load single fiber for specified recontype. 

        Helpful for realtimeplot checks.
        """
        # get testdata directory
        if datatype == 'mems':
            seqdir = '/mems_s_2018111301_axial_0_0_0_01.cs'
        elif datatype == 'angio':
            seqdir = '/ge3d_angio_HD_s_2018072604_HD_01_red8.cs'
        elif datatype == 'gems':
            seqdir = '/gems_s_2018111301_axial90_0_90_0_01.cs'
        elif datatype == 'ge3d':
            seqdir = '/ge3d_s_2018111301_axial_0_0_0_01.cs'
        elif datatpye == 'gems':
            seqdir = '/gems_s_2018111301_axial_0_0_0_01.cs'
        elif datatype == None:
            raise(Exception('Testdata not specified'))
        # get recontype
        if recontype == None:
            raise(Exception('Getting recontype here is not automatic yet'))

        # TODO recontype sanity check

         
        # same as in load_test_cs_data ... 
        testdir = vj.cs+seqdir
        procpar = testdir + '/procpar'
        resultdir = vj.config['testresults_dir']+'/aloha'+seqdir[:-3]+'.nifti'
        savedir = resultdir

        imag = []
        real = []
        imag_orig = []
        real_orig = []

        mask_img = nib.load(glob.glob(testdir+'/*mask.nii.gz')[0])
        affine = mask_img.affine
        mask = mask_img.get_fdata()

        for item in sorted(glob.glob(testdir+'/*kspace*_imag*')):
            data = nib.load(item).get_fdata()
            imag_orig.append(data)
            imag.append(np.multiply(data,mask))
        for item in sorted(glob.glob(testdir+'/*kspace*_real*')):
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
        #
        
        # slicing k-space based on recontype
        if recontype in ['kx-ky','kx-ky_angio']:

            time = kspace_orig.shape[4]//2
            if self.slc == 'offcenter':
                ro = kspace_orig.shape[2]//2-kspace_orig.shape[2]//8
            else:
                ro = kspace_orig.shape[2]//2
            
            fiber_orig = kspace_orig[:,:,ro,:,time]
            fiber_cs = kspace_cs[:,:,ro,:,time]

        elif recontype == 'k-t':

            # take center slice
            if self.slc == 'offcenter':
                slc = kspace_orig.shape[3]//2-kspace_orig.shape[3]//8
                ro = kspace_orig.shape[2]//2-kspace_orig.shape[2]//8
            else:
                slc = kspace_orig.shape[3]//2
                ro = kspace_orig.shape[2]//2
            
            fiber_orig = kspace_orig[:,:,ro,slc,:]
            fiber_cs = kspace_cs[:,:,ro,slc,:]

        elif recontype == 'k':

            # take center slice
            if self.slc == 'offcenter':
                slc = kspace_orig.shape[3]//2-kspace_orig.shape[3]//8
                ro = kspace_orig.shape[2]//2-kspace_orig.shape[2]//8
            else:
                slc = kspace_orig.shape[3]//2
                ro = kspace_orig.shape[2]//2
            
            fiber_orig = kspace_orig[:,:,ro,slc,:]
            fiber_cs = kspace_cs[:,:,ro,slc,:]

        else:
            raise(Exception('this recontype not implemented'))

        return fiber_orig, fiber_cs


    def load_test_cs_data(self,datatype):
        """Load full data (orig, cs, affine, savepath, etc...)"""        

        SLC = self.slc
        SLCNUM = self.slcnum

        if datatype == 'mems':
            seqdir = '/mems_s_2018111301_axial_0_0_0_01.cs'
        elif datatype == 'angio':
            seqdir = '/ge3d_angio_HD_s_2018072604_HD_01_red8.cs'
        elif datatype == 'gems':
            seqdir = '/gems_s_2018111301_axial90_0_90_0_01.cs'
        elif datatype == 'ge3d':
            seqdir = '/ge3d_s_2018111301_axial_0_0_0_01.cs'
        elif datatpye == 'gems':
            seqdir = '/gems_s_2018111301_axial_0_0_0_01.cs'
        elif datatype == None:
            raise(Exception('Testdata not specified'))

        testdir = vj.cs+seqdir
        procpar = testdir + '/procpar'
        resultdir = vj.config['testresults_dir']+'/aloha'+seqdir[:-3]+'.nifti'
        savedir = resultdir

        imag = []
        real = []
        imag_orig = []
        real_orig = []

        mask_img = nib.load(glob.glob(testdir+'/*mask.nii.gz')[0])
        affine = mask_img.affine
        mask = mask_img.get_fdata()

        for item in sorted(glob.glob(testdir+'/*kspace*_imag*')):
            data = nib.load(item).get_fdata()
            imag_orig.append(data)
            imag.append(np.multiply(data,mask))
        for item in sorted(glob.glob(testdir+'/*kspace*_real*')):
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

        if SLC == 'all':
            return (kspace_orig,\
                    kspace_cs,\
                    affine,\
                    procpar,\
                    savedir)
        
        # these are different depending on data
        elif SLC == 'center':
            if datatype in ['mems']:
                center = kspace_orig.shape[vj.config['slc_dim']]//2
                if SLCNUM > 1:
                    num1 = SLCNUM//2
                    num2 = SLCNUM//2
                else:
                    num1 = 1
                    num2 = 0
                return (kspace_orig[...,center-num1:center+num2,:],\
                        kspace_cs[...,center-num1:center+num2,:],\
                        affine,\
                        procpar,\
                        savedir)
            else:
                raise(Exception('not implemented'))
        # if slice and slice number is specified
        else:
            if datatype in ['mems']:
                return (kspace_orig[...,SLC:SLC+SLCNUM,:],\
                        kspace_cs[...,SLC:SLC+SLCNUM,:],\
                        affine,\
                        procpar,\
                        savedir)
            elif datatype in ['angio','ge3d']:
                return (kspace_orig[:,:,SLC:SLC+SLCNUM,:,:],\
                        kspace_cs[:,:,SLC:SLC+SLCNUM,:,:],\
                        affine,\
                        procpar,\
                        savedir)
            elif datatype in ['gems']:
                return (kspace_orig[...,SLC:SLC+SLCNUM,:],\
                        kspace_cs[...,SLC:SLC+SLCNUM,:],\
                        affine,\
                        procpar,\
                        savedir)
            else:
                raise(Exception('not implemented'))

    def save_test_cs_results(self,procpar,\
                                affine,\
                                savedir,\
                                kspace_orig,\
                                kspace_cs,\
                                kspace_filled):
        """Save results after ALOHA in a consistent manner"""

        k_name = ['kspace_orig','kspace_cs','kspace_filled']
        img_name = ['img_orig','img_cs','img_filled']
        dirs = ['orig','zerofilled','filled']
        for num, item in enumerate([kspace_orig,kspace_cs,kspace_filled]):
            recon= vj.recon.ImageSpaceMaker(item, procpar)
            img = recon.make()
            saverecon = vj.io.SaveRecon(procpar,kspace=item,imagespace=img)
            outdir = savedir+'/'+dirs[num]
            saverecon.save(outdir,savetype='full',filetype='nifti')

