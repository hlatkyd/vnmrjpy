import vnmrjpy as vj
import unittest
import glob

class Test_Composer(unittest.TestCase):

    def test_composer(self):

        compdir = vj.config['dataset_dir']+'/parameterfit/composer'
        ref = glob.glob(compdir+'/*composer08_01.fid')[0]
        mems = glob.glob(compdir+'/mems_s_2019022301_01*')[0]
        procpar_ref = ref+'/procpar'
        procpar = mems+'/procpar'
        #make images
        (data_ref, hdr_ref) = vj.io.FidReader(ref).read()
        (data, hdr) = vj.io.FidReader(mems).read()
        kspace_ref = vj.recon.KspaceMaker(data_ref, hdr_ref, procpar_ref).make()
        kspace = vj.recon.KspaceMaker(data, hdr, procpar).make()
        imgspace = vj.recon.ImageSpaceMaker(kspace, procpar).make()
        imgspace_ref = vj.recon.ImageSpaceMaker(kspace_ref, procpar_ref).make()

        #imgspace = vj.util.to_scanner_space(imgspace,procpar)
        #imgspace_ref = vj.util.to_scanner_space(imgspace_ref,procpar_ref)
        
        comp = vj.recon.Composer(imgspace, imgspace_ref, procpar, procpar_ref)
        comp.flirt_registration()
        
