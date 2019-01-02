import unittest
import vnmrjpy as vj
import glob


class Test_FidReader(unittest.TestCase):

    def test_read_gems(self):

        fiddir = sorted(glob.glob(vj.fids+'/gems*'))[0]
        fid = fiddir+'/fid'
        procpar = fiddir+'/procpar'
        reader = vj.io.FidReader(fid)
        data, header = reader.read()
        self.assertEqual(len(data.shape),2)
