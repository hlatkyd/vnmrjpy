import numpy as np
import vnmrjpy as vj
import timeit
import time
import copy

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
        self.conf = vj.config
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


            #----------------------MAIN INIT----------------------------    
            if self.rp['recontype'] == 'k-t':

                slice3d_shape = (self.kspace_cs.shape[self.conf['rcvr_dim']],\
                                self.kspace_cs.shape[self.conf['pe_dim']],\
                                self.kspace_cs.shape[self.conf['et_dim']])

                x_len = self.kspace_cs.shape[self.rp['cs_dim'][0]]
                t_len = self.kspace_cs.shape[self.rp['cs_dim'][1]]
                #each element of weight list is an array of weights in stage s
                weights_list = vj.aloha.make_pyramidal_weights_kxt(\
                                        x_len, t_len, self.rp)
                factors = vj.aloha.make_hankel_decompose_factors(\
                                        slice3d_shape, self.rp)
            
            if self.rp['recontype'] == 'kx-ky_angio':           

                slice3d_shape = (self.kspace_cs.shape[self.conf['rcvr_dim']],\
                                self.kspace_cs.shape[self.conf['pe_dim']],\
                                self.kspace_cs.shape[self.conf['pe2_dim']])

                x_len = self.kspace_cs.shape[self.rp['cs_dim'][0]]
                t_len = self.kspace_cs.shape[self.rp['cs_dim'][1]]
                #each element of weight list is an array of weights in stage s
                weights_list = vj.aloha.make_pyramidal_weights_kxkyangio(\
                                        x_len, t_len, self.rp)
                factors = vj.aloha.make_hankel_decompose_factors(\
                                        slice3d_shape, self.rp)
            
            if self.rp['recontype'] == 'kx-ky_angio':           
            #------------------MAIN ITERATION----------------------------    
            for slc in range(self.kspace_cs.shape[3]):

                for x in range(self.kspace_cs.shape[self.rp['cs_dim'][0]]):

                    slice3d = self.kspace_cs[:,:,x,slc,:]
                    slice3d_orig = copy.deepcopy(slice3d)
                    # main call for solvers
                    slice3d_completed = vj.aloha.pyramidal_solve_kt(slice3d,\
                                                        slice3d_orig,\
                                                        slice3d_shape,\
                                                        weights_list,\
                                                        factors,\
                                                        self.rp)
                    kspace_completed[:,:,x,slc,:] = slice3d_completed
            
                    print('slice {}/{} line {}/{} done.'.format(\
                                slc+1,self.kspace_cs.shape[3],\
                                x+1,self.kspace_cs.shape[2]))

            return kspace_completed

