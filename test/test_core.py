import unittest
import vnmrjpy as vj
import glob
import numpy as np

vj.config['verbose']=False

class Test_Core(unittest.TestCase):

    def test_core_basic(self):

        shape = (128,128,128,5,4)
        data = np.random.rand(*shape)
        varr = vj.varray(data=data)
        self.assertEqual(varr.data.shape,shape)

    def test_core_arrayed_procpar(self):

        gems_arr_te = glob.glob(vj.config['dataset_dir']+'/arrayed/'+\
                            'gems_*array_te*')[0]
        procpar = gems_arr_te+'/procpar'
        pd = vj.read_procpar(procpar)
        varray = vj.read_fid(gems_arr_te)
        #self.assertEqual(varray.arrayed_params)


    def test_read_fid(self):

        fid = glob.glob(vj.config['fids_dir']+'/gems*')[0]
        varr = vj.read_fid(fid)
        self.assertEqual(varr.source,'fid')
        self.assertEqual(len(varr.data.shape),2)
        self.assertEqual(varr.space,'fid')
        self.assertEqual(varr.pd['pslabel'],'gems')
        self.assertEqual(varr.apptype,'im2D')


    def test_read_fdf(self):

        fdf = glob.glob(vj.config['fdfs_dir']+'/gems*')[0]
        varr = vj.read_fdf(fdf)
        self.assertEqual(varr.source,'fdf')
        self.assertEqual(len(varr.data.shape),4)
        self.assertEqual(varr.pd['pslabel'],'gems')
        self.assertEqual(varr.apptype,'im2D')

    def test_read_nifti(self):

        pass
