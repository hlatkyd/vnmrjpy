import numpy as np
import vnmrjpy as vj

class ImageSpaceMaker():
    """ Reconstruct MR images to real space from k-space.

    Generally this is done by fourier transform and corrections.
    Various compressed sensing approaches should be added here
    Hardcoded for each 'seqfil' sequence
    """

    def __init__(self, kspace, procpar):
        """Init params. In Aloha, standard FFT is used when k-space is ready

        Args:
            kspace = np.ndarray([receivers, phase, read, slice, echo])
            procpar = /procpar/file/path

            For CS reconstruction:
            skiptab = 'y' or 'n' procpar parameter
            skipint = procpar parameter to show which k space lines are excluded
        """
        self.kspace_data = kspace
        self.p = vj.io.ProcparReader(procpar).read()
        self.config = vj.config
        
    def make(self):
        """Standard Fourier reconstruction.

        Return imgspace = numpy.ndarray()
        """
        seqfil = str(self.p['seqfil'])
        pe_dim = self.config['pe_dim']
        pe2_dim = self.config['pe2_dim']
        ro_dim = self.config['ro_dim']

        if seqfil in ['gems', 'fsems', 'mems', 'sems', 'mgems']:

            shiftaxes = (self.config['pe_dim'],\
                        self.config['ro_dim'])
            self.kspace_data = np.fft.fftshift(self.kspace_data,axes=shiftaxes) 
            img_space = np.fft.ifft2(self.kspace_data,\
                                    axes=(pe_dim,ro_dim), norm='ortho')
            img_space = np.fft.ifftshift(img_space,\
                                    axes=(pe_dim,ro_dim))
            return img_space

        elif seqfil in ['ge3d','fsems3d','mge3d']:
            
            shiftaxes = (self.config['pe_dim'],\
                        self.config['ro_dim'],\
                        self.config['pe2_dim'])
            self.kspace_data = np.fft.fftshift(self.kspace_data,axes=shiftaxes) 
            img_space = np.fft.ifftn(self.kspace_data,\
                                    axes=(pe_dim,ro_dim,pe2_dim), norm='ortho')
            img_space = np.fft.ifftshift(img_space,\
                                    axes=(pe_dim,ro_dim,pe2_dim))
            return img_space

        elif seqfil in ['ge3d_elliptical']:

            shiftaxes = (self.config['pe_dim'],\
                        self.config['ro_dim'],\
                        self.config['pe2_dim'])
            self.kspace_data = np.fft.fftshift(self.kspace_data,axes=shiftaxes) 
            img_space = np.fft.ifftn(self.kspace_data,\
                                    axes=(pe_dim,ro_dim,pe2_dim), norm='ortho')
            img_space = np.fft.ifftshift(img_space,\
                                    axes=(pe_dim,ro_dim,pe2_dim))
            return img_space

        else:
            raise Exception('Sequence reconstruction not implemented yet')
   

