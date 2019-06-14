import vnmrjpy as vj
import unittest
import numpy as np
import glob

class Test_utils(unittest.TestCase):

    def test_set_procpar(self):

        ppdir = glob.glob(vj.config['dataset_dir']+'/xrecontest/ge*')[0]
        vj.core.utils.set_procpar(ppdir,['rawMG'],['y'])
