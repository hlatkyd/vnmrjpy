import vnmrjpy as vj
import unittest
import numpy as np
import glob

class Test_getpetab(unittest.TestCase):

    def test_getpetab_fromtabfile(self):

        infile = vj.config['tablib_dir']+'/fse128_64_2'
        petab = vj.util.getpetab(infile,is_petab=True)
        self.assertEqual(len(petab.shape),2)

    def test_getpetab_fromprocpar(self):
        infile = glob.glob(vj.fids+'/fsems_s_2018111301_axial*')[0]+'/procpar'
        petab = vj.util.getpetab(infile,is_procpar=True)
        self.assertEqual(len(petab.shape),2)
        
