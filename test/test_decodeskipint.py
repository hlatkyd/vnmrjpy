import unittest
import vnmrjpy as vj
import matplotlib.pyplot as plt
import glob

class Test_DecodeSkipint(unittest.TestCase):

    def test_decode_ge3delliptical(self):

        # check the decoding of original ge3d elliptical skipint

        #fid_dir = sorted(glob.glob(vj.fids+'/ge3d_elliptical_s_2018111301*'))[0]
        fid_dir = sorted(glob.glob(vj.fids+'/ge3d_elliptical_s_20181218*'))[1]
        procpar = fid_dir + '/procpar'
        p = vj.io.ProcparReader(procpar).read()
        skipint = p['skipint']
        decoder = vj.util.SkipintDecoder(p)
        skiptensor = decoder.make_skiptensor()
        plt.imshow(skiptensor)
        plt.show()
