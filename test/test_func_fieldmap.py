import vnmrjpy as vj
import unittest
import numpy as np
import glob
from vnmrjpy.func import concatenate
from vnmrjpy.core.utils import FitViewer3D
from nibabel.viewers import OrthoSlicer3D
import copy

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
    #varr_02 = copy.copy(varr)
    fieldmap = vj.func.make_fieldmap(varr,method='triple_echo', selfmask=False)
    OrthoSlicer3D(fieldmap.data).show()
    # testing fieldmap with single channel receiver
    #varr_02.data = varr_02.data[:,:,:,:,:1,...]
    #fieldmap = vj.func.make_fieldmap(varr_02,method='triple_echo')    
