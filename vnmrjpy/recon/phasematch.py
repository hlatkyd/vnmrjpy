import vnmrjpy as vj
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from nipype.interfaces import fsl

class Composer():
    """Match the phases of different receiver channels

    Ref.: Robinson et.al: Combining Phase Images from  Array Coils Using a
          Short Echo Time Reference Scan (COMPOSER)

    """
    def __init__(self, imgspace, imgspace_ref,procpar, procpar_ref,\
                    rcvrdim='default',workdir=None):
        """
        Args:
            imgspace (np.ndarray) -- complex image data in numpy array
            imgspace_ref (np.ndarray) -- complex reference image data
            procpar

        """
        # moving receiverdim to last, and keeping there from now on
        if rcvrdim == 'default':
            # if input is from default layout switch rcvrdim to dim 4
            self.rcvrdim = 4
            self.imgspace = np.moveaxis(imgspace,[1,2,3,4,0],[0,1,2,3,4])
            self.imgspace_ref = np.moveaxis(imgspace_ref,[1,2,3,4,0],[0,1,2,3,4])
        else:
            raise(Exception('not implemented'))
        self.procpar = procpar
        self.procpar_ref = procpar_ref

        if workdir == None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            #self.workdir = os.path.expanduser('~')+'/tmp_vnmrjpy/cmp'+timestr
            self.workdir = vj.config['fsl_workdir']+'/composer_'+timestr
        else:
            self.workdir = workdir
        # temporary workdir for testing
        self.workdir = '/home/david/dev/vnmrjpy/test/results/composertest'


    def flirt_registration(self, keepfiles=False):
        """

        """
        def _ssos(data):
            """Squared sum of squares from multichannel complex data"""
            ssos = np.sqrt(np.mean(np.absolute(data),axis=self.rcvrdim))
            return ssos

        # make workdir if does not exist
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        # make 6d nifti data for flirt affine transform
        img_6d = vj.util.make_6d(self.imgspace)
        # making niftis
        img_out = self.workdir+'/composer_img_main'
        ref_out = self.workdir+'/composer_ref_main'
        flirt_out = self.workdir+'/composer_flirtout'
        full = self.workdir + '/composer_6d_img'

        writer = vj.io.NiftiWriter(_ssos(self.imgspace), self.procpar,\
                                        input_space='local',\
                                        output_space='scanner')
        writer.write(img_out)
        writer = vj.io.NiftiWriter(_ssos(self.imgspace_ref), self.procpar_ref,\
                                        input_space='local',\
                                        output_space='scanner')
        writer.write(ref_out)
        writer = vj.io.NiftiWriter(img_6d, self.procpar_ref,\
                                        input_space='local',\
                                        output_space='scanner')
        writer.write(full)

        # registering by combined magnitude FSL FLIRT
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = img_out+'.nii.gz'
        flirt.inputs.reference = ref_out+'.nii.gz'
        flirt.inputs.out_file = flirt_out+'.nii.gz'
        flirt.inputs.out_matrix_file = 'composer_invol2refvolmat'
        print(flirt.cmdline)
        res = flirt.run()
        # applying the resulting transform to full data
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = img_out+'.nii.gz'
        flirt.inputs.reference = ref_out+'.nii.gz'
        flirt.inputs.out_file = flirt_out+'.nii.gz'
        flirt.inputs.out_matrix_file = 'composer_invol2refvolmat'
        print(flirt.cmdline)
        vj.io.NiftiReader(flirt_out+'.nii.gz')
        # loading result nifti

        rdr = vj.io.NiftiReader(flirt_out+'.nii.gz')

    def match_from_nifti():
        pass




