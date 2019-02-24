import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt
import unittest
import nibabel as nib
import glob

class Test_transform_to_scanner(unittest.TestCase):

    """
    def test_transform_gems(self):

        axial = glob.glob(vj.fids+'/gems*axial_0_0_0_01.fid')[0]
        axial90 = glob.glob(vj.fids+'/gems*axial90_0_90_0_01.fid')[0]
        coronal = glob.glob(vj.fids+'/gems*coronal_0_0_90_01.fid')[0]
        gems_list = [axial, axial90, coronal]
        gems_procpars = [i+'/procpar' for i in gems_list]
        gems_names = ['axial', 'axial90', 'coronal']
        gems_data_list = []
        for num, item in enumerate(gems_list):
            print(item)
            itemdata, affine = vj.io.FidReader(item).make_image()
            #itemdata = vj.util.to_scanner_space(itemdata,\
            #                                    gems_procpars[num])
            v = nib.viewers.OrthoSlicer3D(itemdata,affine=affine)
            v.show()
            gems_data_list.append(itemdata)

        print(vj.util.check_90deg(gems_procpars[1]))
    """
    def test_transform_mems(self):

        axial = glob.glob(vj.fids+'/mems*axial_0_0_0_01.fid')[0]
        axial90 = glob.glob(vj.fids+'/mems*axial90_0_90_0_01.fid')[0]
        coronal = glob.glob(vj.fids+'/mems*coronal_0_0_90_01.fid')[0]
        mems_list = [axial, axial90, coronal]
        mems_procpars = [i+'/procpar' for i in mems_list]
        mems_names = ['axial', 'axial90', 'coronal']
        mems_data_list = []
        for num, item in enumerate(mems_list):
            print(item)
            itemdata, affine = vj.io.FidReader(item).make_image()
            #itemdata = vj.util.to_scanner_space(itemdata,\
            #                                    mems_procpars[num])
            v = nib.viewers.OrthoSlicer3D(itemdata,affine=affine)
            v.show()
            mems_data_list.append(itemdata)

        print(vj.util.check_90deg(mems_procpars[1]))
        
