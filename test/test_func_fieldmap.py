import vnmrjpy as vj
import unittest
import numpy as np
import glob
from vnmrjpy.func import concatenate

def load_data():

    b0dir = vj.config['dataset_dir'] + '/parameterfit/b0/gems'   
    seqlist = sorted(glob.glob(b0dir+'/gems*'))[4:7]
    print('\nLoading gems data from: \n')
    for i in seqlist:
        print(i)
    
    varr_list = []
    for i in seqlist:
        varr = vj.read_fid(i)
        varr.to_kspace().to_anatomical().to_imagespace()
        varr_list.append(varr)
    
    return concatenate(varr_list)

class Test_fieldmap(unittest.TestCase):

    varr = load_data()
    print('load_data check... ')
    print('varr.pd[te] : {}, varr.data shape {}'.\
        format(varr.pd['te'], varr.data.shape))
    b0map = vj.func.make_fieldmap(varr,method='triple_echo')    
