import vnmrjpy as vj
import numpy as np
import unittest
import nibabel as nib
import os

class Test_T2fitter(unittest.TestCase):

    def test_t2fit(self):

        nifti = vj.niftis+'/mems/mems_20180523_01.nii'
        try:
            procpar = vj.niftis+'/mems/.mems_20180523_01.procpar'
        except:
            procpar = glob.glob(vj.niftis+'/mems/*mems_20180523*')[0]
        # init for output saving
        test_out_dir = vj.config['testresults_dir']+'/fit/t2/'
        if not os.path.exists(test_out_dir):
            os.makedirs(test_out_dir)
        name = 't2map_mems_20180523'
        full_out = test_out_dir+name

        data = nib.load(nifti).get_fdata()
        ppdict = vj.io.ProcparReader(procpar).read()

        fitter = vj.fit.T2Fitter(data, procpar=procpar, automask=True, fitfunc='3param_exp')
        pars = fitter.fit()
        print(len(pars))
        t2map, m0map, const = pars
        print(t2map.shape)
        writer = vj.io.NiftiWriter(t2map,procpar)
        writer.write(full_out)

