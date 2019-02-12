import vnmrjpy as vj
import unittest
import numpy as np
import nibabel as nib
import glob
import copy
import matplotlib.pyplot as plt

class Test_Aloha_gems(unittest.TestCase):

    def test_gems(self):
   
        slc = 'all'  # slice
        slcnum = 'all'  #number of slices
        alohatest = vj.util.AlohaTest(slc,slcnum) 
        kspace_orig, kspace_cs, affine, procpar, savedir = \
                                        alohatest.load_test_cs_data('gems')

        #plt.imshow(np.absolute(kspace_cs[0,:,40,:,0]))
        #plt.show()        
        aloha = vj.aloha.Aloha(kspace_cs,procpar)
        kspace_filled = aloha.recon() 
        alohatest.save_test_cs_results(procpar,affine,savedir,\
                                kspace_orig,kspace_cs,kspace_filled)
