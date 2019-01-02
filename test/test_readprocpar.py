import unittest
import glob
import vnmrjpy as vj

class Test_ProcparReader(unittest.TestCase):

    def test_read_pslabel(self):
        procpar = sorted(glob.glob(vj.fids+'/gems*/procpar'))[0]
        ppr = vj.io.ProcparReader(procpar).read()
        self.assertEqual(ppr['pslabel'],'gems')


