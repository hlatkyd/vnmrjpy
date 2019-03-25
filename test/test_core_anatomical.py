import unittest
import vnmrjpy as vj
import nibabel as nib
import numpy as np
import glob

class Test_anatomical(unittest.TestCase):

    def test_gems_anatomical(self):

        vj.config['verbose'] = False
        vj.config['default_space'] = 'local'
        orients = ['axial','axial90','sag','sag90','cor','cor90']
        orientdir = vj.config['dataset_dir']+'/debug/orient'
        toglob = [orientdir+'/gems*'+i+'_01*' for i in orients] 
        for num in range(len(orients)):
            seq = glob.glob(toglob[num])[0]
            print(seq)
            niftiout = vj.config['testresults_dir']+'/anatomical'
            niftiout = niftiout + '/'+orients[num]

            varr = vj.core.read_fid(seq)
            varr.to_kspace().to_imagespace().to_anatomical()
            vj.core.write.write_nifti(varr,niftiout) 


        
