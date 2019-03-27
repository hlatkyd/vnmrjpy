import vnmrjpy as vj
import unittest
import glob

class Test_Composer(unittest.TestCase):

    def test_composer(self):

        compdir = vj.config['dataset_dir']+'/parameterfit/composer'
        ref = glob.glob(compdir+'/*composer08_01.fid')[0]
        mems = glob.glob(compdir+'/mems_s_2019022301_01*')[0]

        varr = vj.core.read_fid(mems).to_kspace().to_imagespace()
        varr.to_anatomical()
        varr_ref = vj.core.read_fid(ref).to_kspace().to_imagespace()
        varr_ref.to_anatomical()

        test_workdir = vj.config['fsl_workdir']
        varr_matched = vj.func.COMPOSER(varr,varr_ref,\
                            workdir=None,keepfiles=False)
        self.assertEqual(varr_matched.data.shape,varr.data.shape)
