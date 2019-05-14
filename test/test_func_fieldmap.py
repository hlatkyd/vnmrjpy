import vnmrjpy as vj
import unittest

def load_data():

   pass 

class Test_fieldmap(unittest.TestCase):

    def test_triple_echo_gen(self):

    varr = load_data()
    b0map = vj.func.fieldmapgen(varr,method='triple_echo')    
