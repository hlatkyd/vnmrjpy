import vnmrjpy as vj
import unittest

class Test_config(unittest.TestCase):

    def test_configparser(self):

        conf = vj.config
        tol = conf['lmafit_tol']
        self.assertEqual(type(tol),list)
        self.assertEqual(type(vj.config['module_dir']),str)
        self.assertEqual(vj.config['rcvr_dim'],0)
        self.assertEqual(vj.config['vcboost'],bool)
