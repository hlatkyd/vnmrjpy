import numpy as np
import nibabel as nib
import vnmrjpy as vj

class CsTestDataGenerator():
    """Makes .cs bundle for testing
   
    Makes a directory of [seqname].cs, containing:
        imgspace_sum.nii.gz
        kspace_mask.nii.gz
        kspace_imag_ch[n].nii.gz
        kspace_real_ch[n].nii.gz
    """
    def __init__(self,fid,procpar,reduction=4):
        """makes ksapce, imgspace, mask data from fid

        FidReader, KspaceMaker should be available for the sequence
        """    
        p = vj.io.ProcparReader(procpar).read()
        gen = vj.util.SkipintGenerator(procpar=procpar,reduction=reduction)
        self.kspace_mask = gen.generate_kspace_mask()
        fid_data, fid_header = vj.io.FidReader(fid,procpar).read()
        self.kspace = vj.recon.KspaceMaker(fid_data, fid_header, procpar).make()
        self.imgspace = vj.recon.ImageSpaceMaker(self.kspace,procpar).make()
        self.procpar = procpar
        

    def generate(self, savedir=None):
        """saves ksapce, imgspace, mask into savedir"""

        saver = vj.io.SaveRecon(self.procpar,\
                                kspace=self.kspace,\
                                imagespace=self.imgspace)

        masksaver = vj.io.SaveRecon(self.procpar,imagespace=self.kspace_mask)

        if savedir != None:
            saver.save(savedir,savetype='fullkspace') 
            saver.save(savedir,savetype='magn')
            masksaver.save(savedir,savetype='mask')
        else:
            raise(Exception('Please specify output directory'))

