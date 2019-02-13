import vnmrjpy as vj
import numpy as np
import unittest
import nibabel as nib

class Test_T2fitter(unittest.TestCase):

    def test_t2fit(self):

        nifti = vj.niftis+'/mems/mems_20180523_01.nii'
        try:
            procpar = vj.niftis+'/mems/.mems_20180523_01.procpar'
        except:
            procpar = glob.glob(vj.niftis+'/mems/*mems_20180523*')[0]
        data = nib.load(nifti).get_fdata()
        ppdict = vj.io.ProcparReader(procpar).read()

        fitter = vj.fit.T2Fitter(data, procpar=procpar, automask=True)
        t2map = fitter.fit()
        print(t2map.shape)

