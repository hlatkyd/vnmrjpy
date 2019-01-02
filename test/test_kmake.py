import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


class Test_KspaceMaker(unittest.TestCase):

    def test_make_simple2d(self,plot=False,save=True):

        result_dir = vj.config['testresults_dir']+'/kmake'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # standard 2d
        for item in ['/gems*.fid','/sems*.fid','/mgems*.fid','/mems*.fid']:

            fiddir = sorted(glob.glob(vj.fids+item))[0]
            procpar = fiddir+'/procpar'
            fid_data, header = vj.io.FidReader(fiddir).read()
            kspace = vj.recon.KspaceMaker(\
                            fid_data, header, procpar).make()

            self.assertEqual(type(kspace),np.ndarray)
            self.assertEqual(len(kspace.shape),5)
            if plot:
                plt.imshow(np.absolute(\
                        kspace[1,:,:,kspace.shape[3]//2,0]),\
                        cmap='gray',vmax=100)
                plt.show()
            if save:
                vj.io.SaveRecon(procpar, kspace=kspace).save(\
                                result_dir,savetype='fullkspace')

    def test_make_simple3d(self,plot=False,save=True):

        result_dir = vj.config['testresults_dir']+'/kmake'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # standard 3d
        for item in ['/ge3d_s*.fid','/mge3d*.fid']:

            fiddir = sorted(glob.glob(vj.fids+item))[0]
            procpar = fiddir+'/procpar'
            fid_data, header = vj.io.FidReader(fiddir).read()
            kspace = vj.recon.KspaceMaker(\
                            fid_data, header, procpar).make()

            self.assertEqual(type(kspace),np.ndarray)
            self.assertEqual(len(kspace.shape),5)
            if plot:
                plt.imshow(np.absolute(\
                        kspace[1,:,:,kspace.shape[3]//2,0]),\
                        cmap='gray',vmax=100)
                plt.show()
            if save:
                vj.io.SaveRecon(procpar, kspace=kspace).save(\
                                result_dir,savetype='fullkspace')

    def test_make_ge3d_elliptical(self,plot=False,save=True):

        result_dir = vj.config['testresults_dir']+'/kmake'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # standard 3d
        for item in ['/ge3d_elliptical*.fid']:

            fiddir = sorted(glob.glob(vj.fids+item))[0]
            procpar = fiddir+'/procpar'
            fid_data, header = vj.io.FidReader(fiddir).read()
            kspace = vj.recon.KspaceMaker(\
                            fid_data, header, procpar).make()

            self.assertEqual(type(kspace),np.ndarray)
            self.assertEqual(len(kspace.shape),5)
            if plot:
                plt.imshow(np.absolute(\
                        kspace[1,:,:,kspace.shape[3]//2,0]),\
                        cmap='gray',vmax=100)
                plt.show()
            if save:
                vj.io.SaveRecon(procpar, kspace=kspace).save(\
                                result_dir,savetype='fullkspace')
