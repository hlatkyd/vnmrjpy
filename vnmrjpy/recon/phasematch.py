import vnmrjpy as vj
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import nipype as nip

class Composer():
    """Match the phases of different receiver channels

    Ref.: Robinson et.al: Combining Phase Images from  Array Coils Using a
          Short Echo Time Reference Scan (COMPOSER)

    """
    def __init__(self, imgspace, imgspace_ref,
                    procpar, procpar_ref, workdir=None):
        """
        Args:
            imgspace (np.ndarray) -- complex image data in numpy array
            imgspace_ref (np.ndarray) -- complex reference image data
            procpar

        """
        self.imgspace = imgspace
        self.imgspace_ref = imgspace_ref
        self.procpar = procpar
        self.procpar_ref = procpar_ref

        if workdir == None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            self.workdir = os.path.expanduser('~')+'/tmp_vnmrjpy/cmp'+timestr
        else:
            self.workdir = workdir
        # temporary workdir for testing
        self.workdir = '/home/david/dev/vnmrjpy/test/results/composertest'


    def match(self, keepfiles=False):

        def _ssos(data):
            """Squared sum of squares from multichannel complex data"""
            ssos = np.sqrt(np.mean(np.absolute(data),axis=0))
            return ssos

        # make workdir if does not exist
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        # testplot
        """
        plt.subplot(1,2,1)
        plt.imshow(_ssos(self.imgspace)[:,:,13,0])
        plt.subplot(1,2,2)
        plt.imshow(_ssos(self.imgspace_ref)[:,:,18,0])
        plt.show()
        """
        writer = vj.io.NiftiWriter(_ssos(self.imgspace), self.procpar,\
                                    input_space='local',output_space='rat_anatomical')
        writer.write(self.workdir+'/composer_img_main')
        writer = vj.io.NiftiWriter(_ssos(self.imgspace_ref), self.procpar_ref,\
                                    input_space='local',output_space='rat_anatomical')
        writer.write(self.workdir+'/composer_ref_main')

    def match_from_nifti():
        pass
