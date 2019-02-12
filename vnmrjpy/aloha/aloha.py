import numpy as np
import vnmrjpy as vj
import timeit
import time
import copy
import matplotlib.pyplot as plt

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
                        reconpar=None,\
                        check_only=False,\
                        realtimeplot=False):
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
        def _default_filter_size(dims):

            pass        

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
            elif 'gems' in self.p['pslabel']:
                recontype = 'k'
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
                    filter_size = (21,21)
                    cs_dim = (vj.config['pe_dim'],vj.config['pe2_dim'])
                    ro_dim = vj.config['ro_dim']
                    stages = 3
                    fiber_shape = (kspace_shape[vj.config['rcvr_dim']],\
                                    kspace_shape[vj.config['pe_dim']],\
                                    kspace_shape[vj.config['pe2_dim']])
                elif recontype in ['kx-ky_angio']:
                    filter_size = (21,21)
                    cs_dim = (vj.config['pe_dim'],vj.config['pe2_dim'])
                    ro_dim = vj.config['ro_dim']
                    stages = 1
                    fiber_shape = (kspace_shape[vj.config['rcvr_dim']],\
                                    kspace_shape[vj.config['pe_dim']],\
                                    kspace_shape[vj.config['pe2_dim']])
                elif recontype in ['k']:
                    filter_size = 15
                    cs_dim = (vj.config['pe_dim'])
                    ro_dim = vj.config['ro_dim']
                    stages = 3
                    fiber_shape = (kspace_shape[vj.config['rcvr_dim']],\
                                    kspace_shape[vj.config['pe_dim']])
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
        self.weights = vj.aloha.make_kspace_weights(self.rp)
        self.check = check_only
        self.realtimeplot = realtimeplot

    def recon(self):
        """Main reconstruction method for Aloha
        
        Returns:
            kspace_completed (np.ndarray) : full, inferred kspace
        """
        #----------------------------INIT---------------------------------
        def _virtualcoilboost_(data):
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
        # init output data
        if self.rp['virtualcoilboost'] == False:
            kspace_completed = copy.deepcopy(self.kspace_cs)
        elif self.rp['virtualcoilboost'] == True:
            self.kspace_cs = virtualcoilboost_(self.kspace_cs)
            kspace_completed = copy.deepcopy(self.kspace_cs)

        #------------------------------------------------------------------
        #                        1D :    k
        #------------------------------------------------------------------
        if self.rp['recontype'] == 'k':
            print('Processing Aloha reconstruction...')
            #------------------MAIN ITERATION----------------------------    
            time = 0
            for slc in range(self.kspace_cs.shape[3]):

                for ro in range(self.kspace_cs.shape[vj.config['ro_dim']]):

                    fiber3d = self.kspace_cs[:,:,ro,slc,time]
                    fiber3d = vj.aloha.pyramidal_k(fiber3d,\
                                                    self.weights,\
                                                    self.rp)
                    kspace_completed[:,:,ro,slc,time] = fiber3d
            
                    print('ro {}/{} slice {}/{} done.'.format(\
                                ro+1,self.kspace_cs.shape[2],\
                                slc+1,self.kspace_cs.shape[3]))

            return kspace_completed
        #------------------------------------------------------------------
        #                        2D :    k-t
        #------------------------------------------------------------------
        if self.rp['recontype'] in ['k-t']:
            print('Processing Aloha reconstruction...')
            #------------------MAIN ITERATION----------------------------    
            for slc in range(self.kspace_cs.shape[3]):

                for x in range(self.kspace_cs.shape[self.rp['cs_dim'][0]]):

                    # main call for solvers
                    fiber3d = self.kspace_cs[:,:,x,slc,:]
                    fiber3d = vj.aloha.pyramidal_kt(fiber3d,\
                                                self.weights,\
                                                self.rp,\
                                                realtimeplot=self.realtimeplot)
                    kspace_completed[:,:,x,slc,:] = fiber3d
            
                    print('slice {}/{} line {}/{} done.'.format(\
                                slc+1,self.kspace_cs.shape[3],\
                                x+1,self.kspace_cs.shape[2]))

            return kspace_completed
        #TODO reconsider merging angio and kxky
        #------------------------------------------------------------------
        #                     2D :     kx-ky_angio
        #------------------------------------------------------------------
        if self.rp['recontype'] in ['kx-ky_angio']:
            print('Processing Aloha reconstruction...')
            #------------------MAIN ITERATION----------------------------    
            for time in range(self.kspace_cs.shape[4]):
            
                # for testing only
                if self.check:

                    print('Filling center line only')
                    # take center slice
                    slc = self.kspace_cs.shape[self.rp['cs_dim'][1]]//2
                    fiber3d = self.kspace_cs[:,:,slc,:,time]
                    fiber3d_old = copy.copy(fiber3d)
                    fiber3d = vj.aloha.pyramidal_kxky(fiber3d,\
                                                self.weights,\
                                                self.rp,\
                                                realtimeplot=self.realtimeplot)
                    plt.subplot(1,2,1)
                    plt.imshow(np.absolute(fiber3d_old[1,:,:]),\
                                            vmin=0,vmax=50,cmap='gray')
                    plt.subplot(1,2,2)
                    plt.imshow(np.absolute(fiber3d[1,:,:]),\
                                            vmin=0,vmax=50,cmap='gray')
                    plt.show()
            
                    print('slice {}/{} line {}/{} done.'.format(\
                                slc+1,self.kspace_cs.shape[2],\
                                time+1,self.kspace_cs.shape[4]))
                    return

                for slc in range(self.kspace_cs.shape[self.rp['cs_dim'][1]]):

                    fiber3d = self.kspace_cs[:,:,slc,:,time]
                    fiber3d = vj.aloha.pyramidal_kxky(fiber3d,\
                                                    self.weights,\
                                                    self.rp)
                    kspace_completed[:,:,slc,:,time] = fiber3d
            
                    print('slice {}/{} line {}/{} done.'.format(\
                                slc+1,self.kspace_cs.shape[2],\
                                time+1,self.kspace_cs.shape[4]))

            return kspace_completed
        #------------------------------------------------------------------
        #                         2D :    kx-ky
        #------------------------------------------------------------------
        if self.rp['recontype'] in ['kx-ky']:
            print('Processing Aloha reconstruction...')
            #------------------MAIN ITERATION----------------------------    
            for time in range(self.kspace_cs.shape[4]):

                for slc in range(self.kspace_cs.shape[self.rp['cs_dim'][1]]):

                    fiber3d = self.kspace_cs[:,:,slc,:,time]
                    fiber3d = vj.aloha.pyramidal_kxky(fiber3d,\
                                                    self.weights,\
                                                    self.rp)
                    kspace_completed[:,:,slc,:,time] = fiber3d
            
                    print('slice {}/{} line {}/{} done.'.format(\
                                slc+1,self.kspace_cs.shape[2],\
                                time+1,self.kspace_cs.shape[4]))

            return kspace_completed

        #------------------------------------------------------------------
        #                        3D :    kx-ky-t
        #------------------------------------------------------------------
        if self.rp['recontype'] in ['kx-ky-t']:
            raise(Exception('dream on...'))
