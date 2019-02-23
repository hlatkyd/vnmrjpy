import vnmrjpy as vj
import numpy as np
import time
import os

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
            self.workdir = '~/tmp_vnmrjpy/'+timestr
        else:
            self.workdir = workdir

    def match(self, keepfiles=False):

        # make workdir if does not exist
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        print(self.workdir)
        writer = vj.io.NiftiWriter(self.imgspace, self.procpar)
        writer.write(self.workdir+'/composer_img_main')
        writer = vj.io.NiftiWriter(self.imgspace_ref, self.procpar_ref)
        writer.write(self.workdir+'/composer_ref_main')

    
