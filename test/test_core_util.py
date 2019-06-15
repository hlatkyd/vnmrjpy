import vnmrjpy as vj
import unittest
import numpy as np
import glob
import os

class Test_utils(unittest.TestCase):

    def test_change_procpar(self):

        print('Testing change_procpar... ')
        newname = 'testpp'
        ppdir = glob.glob(vj.config['dataset_dir']+'/xrecontest/ge*')[0]
        vj.core.utils.change_procpar(ppdir,['rawMG','rawPH'],['y','y'],\
                        newfile=newname)
        newpp = ppdir+'/'+newname
        pd = vj.read.read_procpar(newpp)
        self.assertEqual(str(pd['rawMG']),'y')
        self.assertEqual(str(pd['rawPH']),'y')
        # delete new procpar
        os.remove(newpp)
