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
        def _get_recontype():
            """get 'recontype', which defines Hankel matrix construction

            Return:
                recontype (str)
            """
            if 'cs' in self.p['apptype']:
                raise(Exception('Cs apptypes not ready yet!'))
            else:
                pass
            # these are for testing. real reconpar will come from apptype
            # TODO
            # or procpar
            if 'angio' in self.p['pslabel']:
                recontype = 'kx-ky_angio'
            elif 'mems' in self.p['pslabel']:
                recontype = 'k-t'
            elif 'ge3d' in self.p['pslabel']:
                recontype = 'kx-ky'
            else:
                raise(Exception('Recon type not implemented yet!'))
            return recontype

        def _get_reconpar(recontype,reconpar,kspace_shape):
            """Make 'rp' recon parameters dictionary

            Reconpar is the main parameter repository of the whole Aloha
            framework. It goes into almost all utility functions during
            recon.
            """

            #TODO maybe make a procpar parameter of this?
            if reconpar == None:

                if recontype == 'k-t':
                    filter_size = (7,5)
                    cs_dim = (vj.config['pe_dim'],vj.config['slc_dim'])
                    ro_dim = vj.config['ro_dim']
                    stages = 3
                    fiber_shape = (kspace_shape[vj.config['rcvr_dim']],\
                                    kspace_shape[vj.config['pe_dim']],\
                                    kspace_shape[vj.config['et_dim']])
                elif recontype in ['kx-ky']:
                    filter_size = (11,11)
                    cs_dim = (vj.config['pe_dim'],vj.config['pe2_dim'])
                    ro_dim = vj.config['ro_dim']
                    stages = 3
                    fiber_shape = (kspace_shape[vj.config['rcvr_dim']],\
                                    kspace_shape[vj.config['pe_dim']],\
                                    kspace_shape[vj.config['pe2_dim']])
                elif recontype in ['kx-ky_angio']:
                    filter_size = (11,11)
                    cs_dim = (vj.config['pe_dim'],vj.config['pe2_dim'])
                    ro_dim = vj.config['ro_dim']
                    stages = 1
                    fiber_shape = (kspace_shape[vj.config['rcvr_dim']],\
                                    kspace_shape[vj.config['pe_dim']],\
                                    kspace_shape[vj.config['pe2_dim']])
                else:
                    raise(Exception('Not implemented'))

                # initialize reconstruction parameters for the whole Aloha
                rp = {'filter_size' : filter_size ,\
                            'cs_dim' : cs_dim ,\
                            'ro_dim' : ro_dim, \
                            'rcvrs' : self.p['rcvrs'].count('y') , \
                            'recontype' : recontype,\
                            'timedim' : vj.config['et_dim'],\
                            'stages' : stages,\
                            'fiber_shape':fiber_shape,\
                            'virtualcoilboost' : vj.config['vcboost'],\
                            'solver' : 'lmafit'}
                return rp
    
            else:
                rp_keys = ['filter_size','cs_dim','ro_dim','rcvrs','recontype',\
                        'timedim','stages','virtualcoilboost','solver']
                # check if dictionary is complete
                for key in rp_keys:
                    if key in reconpar.keys():
                        pass
                    else:
                        raise(Exception('Reconpar key {} not given').format(key))

                return reconpar

        self.p = vj.io.ProcparReader(procpar).read() 
        self.recontype = _get_recontype()
        self.rp = _get_reconpar(self.recontype,reconpar, kspace_cs.shape)
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
            #TODO sort by apptype
            if self.p['apptype'] in ['im2D','im2Dfse','im2Depi']:
                conj_trans_data = np.flip(np.flip(data,axis=\
                            self.conf['pe_dim']),axis=self.conf['ro_dim'])

                conj_trans_data = np.conjugate(conj_trans_data)

            elif self.p['apptype'] in ['im3D','im3Dfse']:
                raise(Exception('not implemented'))

            return np.concatenate((data,conj_trans_data),\
                                axis=self.conf['rcvr_dim'])

        if self.rp['virtualcoilboost'] == False:
            kspace_completed = copy.deepcopy(self.kspace_cs)
        elif self.rp['virtualcoilboost'] == True:
            self.kspace_cs = virtualcoilboost_(self.kspace_cs)
            print(self.kspace_cs.shape)
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
            
            if self.rp['recontype'] in ['kx-ky_angio','kx-ky']:           

                slice3d_shape = (self.kspace_cs.shape[self.conf['rcvr_dim']],\
                                self.kspace_cs.shape[self.conf['pe_dim']],\
                                self.kspace_cs.shape[self.conf['pe2_dim']])

                x_len = self.kspace_cs.shape[self.rp['cs_dim'][0]]
                y_len = self.kspace_cs.shape[self.rp['cs_dim'][1]]
                #each element of weight list is an array of weights in stage s
                if self.rp['recontype'] == 'kx-ky_angio':
                    weights_list = vj.aloha.make_pyramidal_weights_kxkyangio(\
                                        x_len, y_len, self.rp)
                elif self.rp['recontype'] == 'kx-ky':
                    weights_list = vj.aloha.make_pyramidal_weights_kxky(\
                                        x_len, y_len, self.rp)
                else:
                    pass
                factors = vj.aloha.make_hankel_decompose_factors(\
                                        slice3d_shape, self.rp)
            
            #------------------MAIN ITERATION----------------------------    
            for slc in range(self.kspace_cs.shape[3]):

                for x in range(self.kspace_cs.shape[self.rp['cs_dim'][0]]):

                    slice3d = self.kspace_cs[:,:,x,slc,:]
                    slice3d_orig = copy.deepcopy(slice3d)
                    # main call for solvers
                    if self.rp['recontype'] == 'k-t':
                        slice3d_completed = vj.aloha.pyramidal_solve_kt(\
                                                        slice3d,\
                                                        slice3d_orig,\
                                                        slice3d_shape,\
                                                        weights_list,\
                                                        factors,\
                                                        self.rp)
                    elif self.rp['recontype'] in ['kx-ky_angio','kx-ky']:
                        slice3d_completed = vj.aloha.pyramidal_solve_kxky(\
                                                        slice3d,\
                                                        slice3d_orig,\
                                                        slice3d_shape,\
                                                        weights_list,\
                                                        factors,\
                                                        self.rp)
                    else:
                        pass
                    kspace_completed[:,:,x,slc,:] = slice3d_completed
            
                    print('slice {}/{} line {}/{} done.'.format(\
                                slc+1,self.kspace_cs.shape[3],\
                                x+1,self.kspace_cs.shape[2]))

            return kspace_completed

