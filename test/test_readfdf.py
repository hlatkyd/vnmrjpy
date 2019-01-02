import unittest
import vnmrjpy as vj
import glob
import numpy as np
import matplotlib.pyplot as plt

class Test_FdfReader(unittest.TestCase):

    def test_read_img(self):

        fdfdir = sorted(glob.glob(vj.fdfs+'/gems*'))[0]
        
        reader = vj.io.FdfReader(fdfdir)
        data, header = reader.read()
        self.assertEqual(type(data),np.ndarray)
        self.assertEqual(type(header),dict)
        self.assertEqual(len(data.shape),4)
