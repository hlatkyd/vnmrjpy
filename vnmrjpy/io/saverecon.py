import numpy as np
import os
import nibabel as nib
import vnmrjpy as vj
from shutil import copyfile

class SaveRecon():
    """Save reconstruction results to nifti or fdf files in orderly manner

    Args:
        kspace
        imagespace
        procpar

    Methods:

        save
    """
    def __init__(self, procpar, kspace=None, imagespace=None):

        self.imagespace = imagespace
        self.kspace = kspace
        self.ppdict = vj.io.ProcparReader(procpar).read()
        self.procpar = procpar
        self.basename = self.ppdict['pslabel']+'_'+\
                        self.ppdict['comment']

        #raise(Exception('Please specifiy either kspace data or\
        #                    image data, or both'))

    def save(self, outdir, savetype='full',filetype='nifti'):
        """Saves desired files to output dir

        Creates directory, if not present

        Args:
            outdir -- path/to/saved/files
        Options:
            savetype -- determines what to save

                'full' -- combined mangitude,per channel mangitude, phase,
                          per channel kspace real, imag part
                'magn' -- only combined magnitude
                'fullimag'  -- combined magnitude, per channel magnitude,phase
                'fullkspace' -- per channel imag, real kspace
                'mask' -- only save kspace mask

            filetype -- output file type

                'nifti'
                'fdf'
        """
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                raise

        copyfile(self.procpar,outdir+'/procpar')

        if savetype in ['magn','fullimag','full']:
            #saving sum of squares magnitude
            img = np.sum(np.absolute(self.imagespace),axis=0)
            out = outdir+'/'+self.basename+'_magn'
            vj.io.NiftiWriter(img,self.procpar).write(out)
        if savetype in ['fullimag','full']:
            # saving magnitude for each channel
            for ch in range(self.imagespace.shape[0]):
                img = np.absolute(self.imagespace[ch,...])
                out = outdir+'/'+self.basename+'_magn_ch'+str(ch+1)
                vj.io.NiftiWriter(img,self.procpar).write(out)
            # saving phase for each channel
            for ch in range(self.imagespace.shape[0]):
                img = np.arctan2(np.imag(self.imagespace[ch,...]),\
                                np.real(self.imagespace[ch,...]))
                out = outdir+'/'+self.basename+'_phase_ch'+str(ch+1)
                vj.io.NiftiWriter(img,self.procpar).write(out)
        if savetype in ['fullkspace','full']:
            # saving real part of kspace
            for ch in range(self.kspace.shape[0]):
                img = np.real(self.kspace[ch,...])
                out = outdir+'/'+self.basename+'_kspace_ch'+str(ch+1)+'_real'
                vj.io.NiftiWriter(img,self.procpar).write(out)
            # saving imag part of kspace
            for ch in range(self.kspace.shape[0]):
                img = np.imag(self.kspace[ch,...])
                out = outdir+'/'+self.basename+'_kspace_ch'+str(ch+1)+'_imag'
                vj.io.NiftiWriter(img,self.procpar).write(out)
        if savetype == 'full':
            pass
        if savetype == 'mask':
            # saving only mask
            img = self.imagespace
            out = outdir+'/'+self.basename+'_mask'
            vj.io.NiftiWriter(img,self.procpar).write(out)




