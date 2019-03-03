import numpy as np
import nibabel as nib
import vnmrjpy as vj


class NiftiReader():
    """Reads nifti with nibabel, and does some checks


    """
    def __init__(self,nifti,input_space='rat_anatomical'):
        
        self.nifti = nifti
        self.input_space = input_space

    def read():

        n = nib.load(self.nifti)
        header = n.header
        procpar = header['aux_file']
        data = n.get_fdata()
        if procpar == '':
            print('Warning! procpar could not be found based on aux_file')

        return data, header, procpar
