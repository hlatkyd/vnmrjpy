import vnmrjpy as vj
import numpy as np
import unittest
import glob
class Test_Wasabi(unittest.TestCase):

    def test_wasabi(self):

        seq = glob.glob(vj.config['dataset_dir']+\
                        '/parameterfit/wasabi/gems*try1*.fid')[0]
        print(seq)
        varr = vj.core.read_fid(seq).to_kspace()
        varr.to_imagespace().to_anatomical()
        print('varr data shape {}'.format(varr.data.shape))
        print(varr.vdtype)
        ret = vj.func.WASABI(varr) 

