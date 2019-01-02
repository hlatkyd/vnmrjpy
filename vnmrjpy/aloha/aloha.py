import numpy as np
import vnmrjpy as vj
import timeit
import time
import copy

# TESTING PARAMETERS

TESTDIR_ROOT = '/home/david/dev/vnmrjpy/dataset/cs/'
#TESTDIR = TESTDIR_ROOT + 'ge3d_angio_HD_s_2018072604_HD_01.cs'
#TESTDIR = TESTDIR_ROOT + 'gems_s_2018111301_axial_0_0_0_01.cs'
TESTDIR = TESTDIR_ROOT+'mems_s_2018111301_axial_0_0_0_01.cs'
PROCPAR = TESTDIR+'/procpar'
# undersampling dimension
CS_DIM = (1,4)  # phase and slice
RO_DIM = 2
STAGES = 3
FILTER_SIZE = (7,5)

SOLVER = 'ADMM'
# different tolerance for the tages
LMAFIT_TOL_LIST = [5e-2,5e-3,5e-4]
"""
NOTE: testing angio data: rcvrs, phase1, phase2, read
NOTE gems data: 
"""
class Aloha():
    """Aloha framework for Compressed Sensing

    ref: Jin et al.: A general framework for compresed sensing and parallel
        MRI using annihilation filter based low-rank matrix completion (2016)
    
    Process outline:

        1. K-space weighing
        2. pyramidal decomposition
        3. Hankel matrix formation
        4. Low-rank matrix completion
        5. K-space unweighing
    """
    def __init__(self, kspace_cs,\
                        procpar,\
                        kspace_orig=None,\
                        reconpar=None):
        """Aloha parameter initialization

        Args:
            procpar : path to procpar file
            kspace_cs : zerofilled cs kspace in numpy array
            reconpar: dictionary, ALOHA recon parameters
                    keys:
                        filter_size
                        rcvrs
                        cs_dim
                        recontype

        TODO : kspace_orig is for test purposes only
        """
        def _get_recontype(reconpar):

            if 'angio' in self.p['pslabel']:
                recontype = 'kx-ky_angio'
            elif 'mems' in self.p['pslabel']:
                recontype = 'k-t'
            else:
                raise(Exception('Recon type not implemented yet!'))
            return recontype

        def _get_reconpar(recontype):
            """Make 'rp' recon parameters dictionary
            """
            filter_size = (7,5)
            if recontype == 'k-t':
                cs_dim = (vj.config['pe_dim'],vj.config['slc_dim'])
                ro_dim = vj.config['ro_dim']
                stages = 3
            elif recontype in ['kx-ky', 'kx-ky_angio']:
                raise(Exception('Not implemented'))
            rp = {'filter_size' : filter_size ,\
                        'cs_dim' : cs_dim ,\
                        'ro_dim' : ro_dim, \
                        'rcvrs' : self.p['rcvrs'].count('y') , \
                        'recontype' : recontype,\
                        'timedim' : vj.config['et_dim'],\
                        'stages' : stages,\
                        'virtualcoilboost' : vj.config['vcboost'],\
                        'solver' : 'lmafit'}
            return rp

        self.p = vj.io.ProcparReader(procpar).read() 
        self.recontype = _get_recontype(reconpar)
        self.rp = _get_reconpar(self.recontype)
        self.kspace_cs = np.array(kspace_cs, dtype='complex64')

    def recon(self):
        """Main reconstruction method for Aloha
        
        Returns:
            kspace_completed (np.ndarray) : full, inferred kspace
        """
        #----------------------------INIT---------------------------------
        def virtualcoilboost_(data):
            """virtual coil data augmentation"""

            return boosted

        if self.rp['virtualcoilboost'] == False:
            kspace_completed = copy.deepcopy(self.kspace_cs)
        elif self.rp['virtualcoilboost'] == True:
            self.kspace_cs = virtualcoilboost_(self.kspace_cs)
            kspace_completed = copy.deepcopy(self.kspace_cs)

        #------------------------------------------------------------------
        #           2D :    k-t ; kx-ky ; kx-ky_angio
        #------------------------------------------------------------------
        if self.rp['recontype'] in ['k-t','kx-ky','kx-ky_angio']:


            if self.rp['recontype'] == 'k-t':
            #----------------------MAIN INIT----------------------------    
                slice3d_shape = (self.kspace_cs.shape[0],\
                                self.kspace_cs.shape[1],\
                                self.kspace_cs.shape[4])

                x_len = kspace_cs.shape[self.rp['cs_dim'][0]]
                t_len = kspace_cs.shape[self.rp['cs_dim'][1]]
                #each element of weight list is an array of weights in stage s
                weights_list = vj.aloha.make_pyramidal_weights_kxt(\
                                        x_len, t_len, self.rp)
                factors = vj.aloha.make_hankel_decompose_factors(\
                                        slice3d_shape, self.rp)
            
            #print('factors len : {}'.format(len(factors)))
            #print('factors[0] shape : {}'.format(factors[0][0].shape))
           
            #------------------MAIN ITERATION----------------------------    
            for slc in range(self.kspace_cs.shape[3]):

                for x in range(self.kspace_cs.shape[self.rp['cs_dim'][0]]):
                    #TODO plotind is for testing only delete afterward
                    if x == self.kspace_cs.shape[self.rp['cs_dim'][0]]//2-5:
                        plotind = 1
                    else: 
                        plotind = 0

                    slice3d = self.kspace_cs[:,:,x,slc,:]
                    slice3d_orig = copy.deepcopy(slice3d)
                    slice3d_completed = vj.aloha.pyramidal_solve_kt(slice3d,\
                                                        slice3d_orig,\
                                                        slice3d_shape,\
                                                        weights_list,\
                                                        factors,\
                                                        self.rp)
                    kspace_completed[:,:,x,slc,:] = slice3d_completed
            
                    print('slice {}/{} line {}/{} done.'.format(\
                                slc+1,kspace_cs.shape[3],x+1,kspace_cs.shape[2]))

            return kspace_completed

#----------------------------------FOR TESTING---------------------------------

def save_test_data(kspace_orig, kspace_cs, kspace_filled, affine):

    def makeimg(kspace):

        img_space = np.fft.ifft2(kspace, axes=(1,2), norm='ortho')
        img_space = np.fft.fftshift(img_space, axes=(1,2))
        return img_space

    SAVEDIR = '/home/david/dev/vnmrjpy/aloha/result_aloha/'

    # saving kspace
    kspace_orig_ch1 = nib.Nifti1Image(np.absolute(kspace_orig[0,...]), affine) 
    nib.save(kspace_orig_ch1, SAVEDIR+'kspace_orig')
    kspace_cs_ch1 = nib.Nifti1Image(np.absolute(kspace_cs[0,...]), affine) 
    nib.save(kspace_cs_ch1, SAVEDIR+'kspace_cs')
    kspace_filled_ch1 = nib.Nifti1Image(np.absolute(kspace_filled[0,...]),\
                                         affine) 
    nib.save(kspace_filled_ch1, SAVEDIR+'kspace_filled')
    # saving 6D raw kspace - 5D standard, 6th real/imag
    kspace_filled_6d_data = np.stack((np.real(kspace_filled), \
                            np.imag(kspace_filled)), \
                            axis=len(kspace_filled.shape))
    kspace_filled_6d = nib.Nifti1Image(kspace_filled_6d_data,affine)
    nib.save(kspace_filled_6d, SAVEDIR+'kspace_filled_6d')
    
    


    imgspace_orig = makeimg(kspace_orig) 
    imgspace_cs = makeimg(kspace_cs) 
    imgspace_filled = makeimg(kspace_filled) 
    # saving combined magnitude
    name_list = ['img_orig_full','img_cs_full','img_filled_full']
    for num,item in enumerate([imgspace_orig,imgspace_cs,imgspace_filled]):
        img_comb = np.mean(np.absolute(item),axis=0)
        img_comb = nib.Nifti1Image(img_comb,affine)
        nib.save(img_comb,SAVEDIR+name_list[num])
    # saving for fslview
    imgspace_orig_ch1 = nib.Nifti1Image(np.absolute(imgspace_orig[0,...]),\
                                        affine) 
    nib.save(imgspace_orig_ch1, SAVEDIR+'imgspace_orig')
    imgspace_cs_ch1 = nib.Nifti1Image(np.absolute(imgspace_cs[0,...]),\
                                        affine) 
    nib.save(imgspace_cs_ch1, SAVEDIR+'imgspace_cs')
    imgspace_filled_ch1 = nib.Nifti1Image(np.absolute(imgspace_filled[0,...]),\
                                        affine) 
    nib.save(imgspace_filled_ch1, SAVEDIR+'imgspace_filled')

    #saving 5d magnitude and phase
    imgspace_filled_5d_magn = np.absolute(imgspace_filled)
    imgspace_filled_5d_phase = np.arctan2(np.imag(imgspace_filled),\
                                        np.real(imgspace_filled))
    magn5d = nib.Nifti1Image(imgspace_filled_5d_magn,affine)
    phase5d = nib.Nifti1Image(imgspace_filled_5d_phase,affine)
    nib.save(magn5d,SAVEDIR+'img_filled_5d_magn')
    nib.save(phase5d,SAVEDIR+'img_filled_5d_phase')


if __name__ == '__main__':

    kspace_orig, kspace_cs, affine = load_test_data()

    aloha = ALOHA(PROCPAR, kspace_cs, kspace_orig)
    start_time = time.time()
    kspace_filled = aloha.recon()
    print('elapsed time {}'.format(time.time()-start_time))

    save_test_data(kspace_orig, kspace_cs, kspace_filled, affine)
