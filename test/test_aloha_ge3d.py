import vnmrjpy as vj
import unittest
import numpy as np
import nibabel as nib
import glob
import copy
import matplotlib.pyplot as plt

# test on small number of slices only for 'speed'
SLC = 'all'
SLCNUM = 1

class Test_Aloha_ge3d(unittest.TestCase):

    def test_ge3d(self):
   
        slc = SLC  # slice
        slcnum = SLCNUM  #number of slices
        alohatest = vj.util.AlohaTest(slc,slcnum) 
        kspace_orig, kspace_cs, affine, procpar, savedir = \
                                        alohatest.load_test_cs_data('ge3d')

        #plt.imshow(np.absolute(kspace_cs[0,:,40,:,0]))
        #plt.show()        
        aloha = vj.aloha.Aloha(kspace_cs,procpar)
        kspace_filled = aloha.recon() 
        alohatest.save_test_cs_results(procpar,affine,savedir,\
                                kspace_orig,kspace_cs,kspace_filled)
