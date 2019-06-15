import vnmrjpy as vj
import copy
import numpy as np
from functools import reduce
from vnmrjpy.core.utils import vprint
import warnings
import matplotlib.pyplot as plt

"""
vnmrjpy.varray
==============

Contaions the main varray object, and the basic k-space and image space
building methods as well as conversion between the two.

"""
class varray():
    """Main vnmrjpy object to carry data and reconstruction information.

    Attributes:
        
    Methods:


    """
    def __init__(self, data=None, pd=None, source=None, is_zerofilled=False,\
                is_kspace_complete=False, fid_header=None, fdf_header=None,\
                nifti_header=None, space=None, intent=None,dtype=None,\
                arrayed_params=[None,1], seqcon=None, apptype=None,\
                vdtype=None, sdims=None, dims=None, description=None,\
                fid_path=None):

        # data stored in numpy nd.array
        self.data = data
        # procpar dictionary
        self.pd = pd
        # file type varray was made from, eg: fid, fdf, nift
        self.source = source
        #TODO whats this again???
        self.intent = intent
        # coordinate space, anatomical, local, global
        self.space = space
        # dtype in data
        self.dtype= dtype
        # varian data type, 'imagespace' or 'kspace' or 'fid'
        self.vdtype = vdtype
        # if kspace is zerofilled
        self.is_zerofilled = is_zerofilled
        # if kspace is complete
        self.is_kspace_complete = is_kspace_complete
        # path to fid directory
        self.fid_path = fid_path
        # fid header dictionary if source is fid
        self.fid_header = fid_header
        # fdf dictionary if source is fdf
        self.fdf_header = fid_header
        # nifti header, can be constructed almost anytime
        self.nifti_header = nifti_header
        # which parameters were arrayed "vnmrj style"
        self.arrayed_params = arrayed_params
        # These are for convenience, can also be found in procpar dict
        # seqcon parameter
        self.seqcon = seqcon
        # apptype parameter
        self.apptype = apptype
        # spatial dimensions
        # eg.: [read, phase, slice, time, rcvr]
        self.sdims = sdims
        # gradient coding dimensions
        # eg.: v.dims = [x,y,z,t,rcvr]
        self.dims = dims
        # optional string
        self.description = description
    
    def flip_axis(self,axis):
        """Flip data on axis 'x','y','z' or 'phase','read','slice'"""
        return vj.core.transform._flip_axis(self,axis)        

    def set_nifti_header(self):

        return vj.core.niftitools._set_nifti_header(self)

    def set_dims(self):
        """Set dims, meaning x,y,z axes order""" 
        return vj.core.transform._set_dims(self)

    # put the transforms into another file for better visibility

    def to_local(self):
        """Transform to original space at k-space formation"""
        return vj.core.transform._to_local(self)

    def to_scanner(self):
        """Transform data to scanner coordinate space by properly swapping axes

        Standard vnmrj orientation - meaning rotations are 0,0,0 - is axial, with
        x,y,z axes (as global gradient axes) corresponding to phase, readout, slice.
        vnmrjpy defaults to handling numpy arrays of:
                    (readout, phase, slice/phase2, time*echo, receivers)
        but arrays of
                            (x,y,z, time*echo, reveivers)
        is also desirable in some cases (for example registration in FSL flirt,
        or third party stuff)

        Euler angles of rotations are psi, phi, theta.

        Also corrects reversed X gradient and sliceorder
        """
        if vj.core.transform._check_90deg(self.pd):
            pass
        else:
            raise(Exception('Only Euler angles of 0,90,180 are permitted.'))    

        return vj.core.transform._to_scanner(self)

    def to_anatomical(self):
        """Transform to rat brain anatomical coordinate system, the 'usual'"""
        return vj.core.transform._to_anatomical(self)

    def to_global(self):
        """Fixed to scanner, but enlarges FOV if necessary to support oblique.
        """
        return vj.core.transform._to_global(self)

    def to_kspace(self, raw=False, zerofill=True,
                    method='vnmrjpy',
                    epiref_type='default',epinav='default'):
        """Build the k-space from the raw fid data and procpar.

        Raw fid_data is numpy.ndarray(blocks, traces * np) format. Should be
        untangled based on 'seqcon' or 'seqfil' parameters.
        seqcon chars refer to (echo, slice, Pe1, Pe2, Pe3)

        For compatibility, an interface to Xrecon is provided. Set 'method' to 
        'xrecon'
        Args:
            raw
            zerofill
            method -- 'xrecon', or 'vnmrjpy'


        note:
        PREVIOUS:
                ([rcvrs, phase, read, slice, echo*time])
        NOW:
                ([phase, read, slice, echo*time, rcvrs])
        """
        # ====================== Child functions, helpers======================
        def _is_interleaved(ppdict):
            res  = (int(ppdict['sliceorder']) == 1)
            return res
        def _is_evenslices(ppdict):
            try:
                res = (int(ppdict['ns']) % 2 == 0)
            except:
                res = (int(ppdict['pss']) % 2 == 0)
            return res
        def make_im2D():
            """Child method of 'make', provides the same as vnmrj im2Drecon"""
            p = self.pd 
            rcvrs = int(p['rcvrs'].count('y'))
            (read, phase, slices) = (int(p['np'])//2,int(p['nv']),int(p['ns']))
            if 'ne' in p.keys():
                echo = int(p['ne'])
            else:
                echo = 1
            time = 1
            # this is the old shape which worked, better to reshape at the end
            finalshape = (rcvrs, phase, read, slices,echo*time*array_length)
            final_kspace = np.zeros(finalshape,dtype='complex64')
           
            for i in range(array_length):
 
                kspace = self.data[i*blocks:(i+1)*blocks,...]

                if p['seqcon'] == 'nccnn':
                    shape = (rcvrs, phase, slices, echo*time, read)
                    kspace = np.reshape(kspace, shape, order='C')
                    kspace = np.moveaxis(kspace, [0,1,4,2,3], [0,1,2,3,4])
                    

                elif p['seqcon'] == 'nscnn':

                    raise(Exception('not implemented'))

                elif p['seqcon'] == 'ncsnn':

                    preshape = (rcvrs, phase, slices*echo*time*read)
                    shape = (rcvrs, phase, slices, echo*time, read)
                    kspace = np.reshape(kspace, preshape, order='F')
                    kspace = np.reshape(kspace, shape, order='C')
                    kspace = np.moveaxis(kspace, [0,1,4,2,3], [0,1,2,3,4])

                elif p['seqcon'] == 'ccsnn':

                    preshape = (rcvrs, phase, slices*echo*time*read)
                    shape = (rcvrs, phase, slices, echo*time, read)
                    kspace = np.reshape(kspace, preshape, order='F')
                    kspace = np.reshape(kspace, shape, order='C')
                    kspace = np.moveaxis(kspace, [0,1,4,2,3], [0,1,2,3,4])
                else:
                    raise(Exception('Not implemented yet'))
                if _is_interleaved(p): # 1 if interleaved slices
                    if _is_evenslices(p):
                        c = np.zeros(kspace.shape, dtype='complex64')
                        c[...,0::2,:] = kspace[...,:slices//2,:]
                        c[...,1::2,:] = kspace[...,slices//2:,:]
                        kspace = c
                    else:
                        c = np.zeros(kspace.shape, dtype='complex64')
                        c[...,0::2,:] = kspace[...,:(slices+1)//2,:]
                        c[...,1::2,:] = kspace[...,(slices-1)//2+1:,:]
                        kspace = c

                final_kspace[...,i*echo*time:(i+1)*echo*time] = kspace

            self.data = final_kspace
            # additional reordering
            self.data = np.moveaxis(self.data,[0,1,2,3,4],[4,1,0,2,3])
            # swap axes 0 and 1 so phase, readout etc is the final order
            self.data = np.swapaxes(self.data,0,1)
            return self

        def make_im2Dcs(**kwargs):
            """
            These (*cs) are compressed sensing variants
            """

            def decode_skipint_2D(skipint):

               pass
 
            raise(Exception('not implemented'))

        def make_im2Depi(**kwargs):

            p = self.pd
            # count navigator echos, also there is a unused one
            if p['navigator'] == 'y':
                pluspe = 1 + int(p['nnav'])  # navigator echo + unused
            else:
                pluspe = 1  # unused only
            
            # init main params
            # -------------------------------------------------------
            comp_seg = p['cseg']
            altread = p['altread']
            nseg = int(p['nseg'])  # number of segments
            etl = int(p['etl'])  # echo train length
            kzero = int(p['kzero'])  
            images = int(p['images'])  # repetitions
            rcvrs = int(p['rcvrs'].count('y'))
            time = len(p['image'])  # total volumes including references
            npe = etl + pluspe  # total phase encode lines per shot
            # getting phase encode scheme
            if p['pescheme'] == 'l':
                pescheme = 'linear'
            elif p['pescheme'] == 'c':
                pescheme = 'centric'
            else:
                pescheme = None
            #TODO make petable?
            if p['petable'] == 'y':
                petab_file = None #TODO
                phase_order = vj.core.epitools.\
                            _get_phaseorder_frompetab(petab_file)
            else:
                phase_order = vj.core.epitools.\
                            _get_phaseorder_frompar(nseg,npe,etl,kzero)

            # init final shape
            if int(p['pro']) != 0:
                (read, phase, slices) = (int(p['nread']), \
                                            int(p['nphase']), \
                                            int(p['ns']))
            else:
                (read, phase, slices) = (int(p['nread'])//2, \
                                            int(p['nphase']), \
                                            int(p['ns']))
            finalshape = (read, phase, slices,time*array_length, rcvrs)
            final_kspace = np.zeros(finalshape,dtype='complex64')
            #navshape = (rcvrs, int(p['nnav']),read,slices,
            #        echo*time*array_length)
            #nav = np.zeros(navshape,dtype='complex64')  #full set of nav echos

            # sanity check
            if int(self.fid_header['np']) != int((etl +pluspe)*read*2):
                raise Exception("np and kspace format doesn't match")

            for i in range(array_length):
 
                # arrange to kspace, but don't do corrections
                kspace = self.data[i*blocks:(i+1)*blocks,...]

                # this case repetitions are in different blocks
                if p['seqcon'] == 'ncnnn':
                
                    #preshape = (rcvrs, time, nseg, slices, npe, read)
                    preshape = (time, rcvrs, nseg, slices, npe, read)
                    kspace = np.reshape(kspace, preshape, order='c')
                    # utility swaps...
                    kspace = np.swapaxes(kspace, 2,3)
                    kspace = np.swapaxes(kspace, 0,1)
                    # dims now: [rcvrs,time,nslices, nseg, phase, read]
                    # reverse odd readout lines
                    kspace = vj.core.epitools._reverse_odd(kspace,\
                                                read_dim=5,phase_dim=4)
                    # correct reversed echos for main ghost corr
                    kspace = vj.core.epitools._navigator_scan_correct(kspace,p)
                    # navigator correct
                    # this is for intersegment, and additional ghost corr
                    kspace = vj.core.epitools.\
                            _navigator_echo_correct(kspace,npe,etl,method='single')
                    # remove navigator echos 
                    kspace = vj.core.epitools._remove_navigator_echos(kspace,etl)
                    kspace = vj.core.epitools._zerofill(kspace, phase, nseg)
                    # start combining segments
                    kspace = vj.core.epitools._combine_segments(kspace,pescheme)
                    # reshape to [read,phase,slice,time,rcvrs]
                    kspace = vj.core.epitools._reshape_stdepi(kspace)
                    # correct for interleaved slices
                    kspace = vj.core.epitools._correct_ilepi(kspace,p)
                    kspace = vj.core.epitools._refcorrect(\
                                        kspace,p,method=epiref_type)

                else:
                    raise(Exception('This seqcon not implemented in epip'))
                # -------------------epi kspace preprocessing------------------
                # TODO check 'image' after array merginf
                final_kspace[...,i*time:(i+1)*time,:] = kspace
            # --------------------- kspace finished----------------------------
            self.data = final_kspace
            return self

        def make_im2Depics():
            raise(Exception('not implemented'))
        def make_im2Dfse(**kwargs):

            p = self.pd
            #petab = vj.util.getpetab(self.procpar,is_procpar=True)
            petab = vj.core.read_petab(self.pd)
            nseg = int(p['nseg'])  # seqgments
            etl = int(p['etl'])  # echo train length
            kzero = int(p['kzero'])  
            images = int(p['images'])  # repetitions
            (read, phase, slices) = (int(p['np'])//2,int(p['nv']),int(p['ns']))

            # setting time params
            echo = 1
            time = images

            phase_sort_order = np.reshape(np.array(petab),petab.size,order='C')
            # shift to positive
            phase_sort_order = phase_sort_order + phase_sort_order.size//2-1

            finalshape = (rcvrs, phase, read, slices,echo*time*array_length)
            final_kspace = np.zeros(finalshape,dtype='complex64')
           
            for i in range(array_length):
 
                kspace = self.data[i*blocks:(i+1)*blocks,...]
                if p['seqcon'] == 'nccnn':

                    #TODO check for images > 1
                    preshape = (rcvrs, phase//etl, slices, echo*time, etl, read)
                    shape = (rcvrs, echo*time, slices, phase, read)
                    kspace = np.reshape(kspace, preshape, order='C')
                    kspace = np.swapaxes(kspace,1,3)
                    kspace = np.reshape(kspace, shape, order='C')
                    # shape is [rcvrs, phase, slices, echo*time, read]
                    kspace = np.swapaxes(kspace,1,3)
                    kspace_fin = np.zeros_like(kspace)
                    kspace_fin[:,phase_sort_order,:,:,:] = kspace
                    kspace_fin = np.moveaxis(kspace_fin, [0,1,4,2,3], [0,1,2,3,4])
                    kspace = kspace_fin
                else:
                    raise(Exception('not implemented'))
                
                if _is_interleaved(p): # 1 if interleaved slices
                    if _is_evenslices(p):
                        c = np.zeros(kspace.shape, dtype='complex64')
                        c[...,0::2,:] = kspace[...,:slices//2,:]
                        c[...,1::2,:] = kspace[...,slices//2:,:]
                        kspace = c
                    else:
                        c = np.zeros(kspace.shape, dtype='complex64')
                        c[...,0::2,:] = kspace[...,:(slices+1)//2,:]
                        c[...,1::2,:] = kspace[...,(slices-1)//2+1:,:]
                        kspace = c
            
                final_kspace[...,i*echo*time:(i+1)*echo*time] = kspace

            self.data = final_kspace
            # additional reordering
            self.data = np.moveaxis(self.data,[0,1,2,3,4],[4,1,0,2,3])
            # swap axes 0 and 1 so phase, readout etc is the final order
            self.data = np.swapaxes(self.data,0,1)
            return self

        def make_im2Dfsecs(**kwargs):
            raise(Exception('not implemented'))
        def make_im3D(**kwargs):
            """Child method of 'make', provides the same as vnmrj im3Drecon"""
            p = self.pd 
            rcvrs = int(p['rcvrs'].count('y'))
            (read, phase, phase2) = (int(p['np'])//2,int(p['nv']),int(p['nv2']))
            if 'ne' in p.keys():
                echo = int(p['ne'])
            else:
                echo = 1
            if 'images' in p.keys():
                time = int(p['images'])
            else:
                time = 1

            finalshape = (rcvrs, phase, read, phase2,echo*time*array_length)
            final_kspace = np.zeros(finalshape,dtype='complex64')
           
            for i in range(array_length):
 
                kspace = self.data[i*blocks:(i+1)*blocks,...]

                if p['seqcon'] == 'nccsn':
                
                    preshape = (rcvrs,phase2,phase*echo*time*read)
                    shape = (rcvrs,phase2,phase,echo*time,read)
                    kspace = np.reshape(kspace,preshape,order='F')
                    kspace = np.reshape(kspace,shape,order='C')
                    kspace = np.moveaxis(kspace, [0,2,4,1,3], [0,1,2,3,4])
                    # what is this??
                    #kspace = np.flip(kspace,axis=3)

                if p['seqcon'] == 'ncccn':
                    preshape = (rcvrs,phase2,phase*echo*time*read)
                    shape = (rcvrs,phase,phase2,echo*time,read)
                    kspace = np.reshape(kspace,preshape,order='F')
                    kspace = np.reshape(kspace,shape,order='C')
                    kspace = np.moveaxis(kspace, [0,2,4,1,3], [0,1,2,3,4])
        
                if p['seqcon'] == 'cccsn':
                
                    preshape = (rcvrs,phase2,phase*echo*time*read)
                    shape = (rcvrs,phase,phase2,echo*time,read)
                    kspace = np.reshape(kspace,preshape,order='F')
                    kspace = np.reshape(kspace,shape,order='C')
                    kspace = np.moveaxis(kspace, [0,2,4,1,3], [0,1,2,3,4])

                if p['seqcon'] == 'ccccn':
                    
                    shape = (rcvrs,phase2,phase,echo*time,read)
                    kspace = np.reshape(kspace,shape,order='C')
                    kspace = np.moveaxis(kspace, [0,2,4,1,3], [0,1,2,3,4])

                final_kspace[...,i*echo*time:(i+1)*echo*time] = kspace

            self.data = final_kspace
            # additional reordering
            self.data = np.moveaxis(self.data,[0,1,2,3,4],[4,1,0,2,3])
            # swap axes 0 and 1 so phase, readout etc is the final order
            self.data = np.swapaxes(self.data,0,1)
            
            return self

        def make_im3Dcs():
            """
            3D compressed sensing
            sequences : ge3d, mge3d, se3d, etc
            """
            # -------------------im3Dcs Make helper functions ---------------------

            def decode_skipint_3D(skipint):
                """
                Takes 'skipint' parameter and returns a 0-1 matrix according to it
                which tells what lines are acquired in the phase1-phase2 plane
                """
                BITS = 32  # Skipint parameter is 32 bit encoded binary, see spinsights
                skip_matrix = np.zeros([int(p['nv']), int(p['nv2'])])
                skipint = [int(x) for x in skipint]
                skipint_bin_vals = [str(np.binary_repr(d, BITS)) for d in skipint]
                skipint_bin_vals = ''.join(skipint_bin_vals)
                skipint_bin_array = np.asarray([int(i) for i in skipint_bin_vals])
                skip_matrix = np.reshape(skipint_bin_array, skip_matrix.shape)

                return skip_matrix

            def fill_kspace_3D(pre_kspace, skip_matrix, shape):
                """
                Fills up reduced kspace with zeros according to skip_matrix
                returns zerofilled kspace in the final shape
                """
                kspace = np.zeros(shape, dtype=complex)
                if self.p['seqcon'] == 'ncccn':

                    n = int(self.p['nv'])
                    count = 0
                    for i in range(skip_matrix.shape[0]):
                        for k in range(skip_matrix.shape[1]):
                            if skip_matrix[i,k] == 1:
                                kspace[:,i,k,:,:] = pre_kspace[:,count,:,:]
                                count = count+1
                self.kspace = kspace
                return kspace

            #------------------------im3Dcs make start -------------------------------

            kspace = self.pre_kspace
            p = self.p
            (read, phase, phase2) = (int(p['np'])//2, \
                                    int(p['nv']), \
                                     int(p['nv2']))

            shiftaxis = (self.config['pe_dim'],\
                        self.config['ro_dim'],\
                        self.config['pe2_dim'])

            if 'ne' in p.keys():
                echo = int(p['ne'])
            else:
                echo = 1

            time = 1

            if p['seqcon'] == 'nccsn':

                pass

            if p['seqcon'] == 'ncccn':
            
                skip_matrix = decode_skipint_3D(p['skipint'])
                pre_phase = int(self.fid_header['ntraces'])    
                shape = (self.rcvrs, phase, phase2, echo*time, read)
                pre_shape = (self.rcvrs, pre_phase, echo*time, read)
                pre_kspace = np.reshape(kspace, pre_shape, order='c')
                kspace = fill_kspace_3D(pre_kspace, skip_matrix, shape)
                kspace = np.moveaxis(kspace, [0,1,4,2,3],[0,1,2,3,4])
                
            self.kspace = kspace
            return kspace

        def make_im3Dute():
            raise(Exception('not implemented')) 

        # ========================== INIT =====================================
        # if data is not from fid, just fft it
        if self.vdtype == 'imagespace':
            raise(Exception('not implemented yet'))        
            #TODO something like this:

            #self.data = vj.core.recon._fft(self.data,dims)
            #self.vdype = 'kspace'                

        #=========================== Xrecon ===================================
        vprint(' making seqfil : {}'.format(self.pd['seqfil']))

        if method == 'xrecon':
            self = vj.xrecon.make_temp_dir(self)
            self = vj.xrecon.mod_procpar(self,output='kspace')
            self = vj.xrecon.call(self)
            self = vj.xrecon.loadfdf(self)
            self = vj.xrecon.clean(self)
            return self

        # ========================= Vnmrjpy recon ============================
        elif method == 'vnmrjpy':

            # check if data is really from fid
            if self.vdtype is not 'fid':
                raise(Exception('varray data is not fid data.'))
            self.data = np.vectorize(complex)(self.data[:,0::2],\
                                                    self.data[:,1::2])
            # check for arrayed parameters, save the length for later 
            array_length = reduce(lambda x,y: x*y, \
                            [i[1] for i in self.arrayed_params])
            blocks = self.data.shape[0] // array_length
            vprint('Making k-space for '+ str(self.apptype)+' '\
                    +str(self.pd['seqfil'])+' seqcon: '+str(self.pd['seqcon']))
            rcvrs = self.pd['rcvrs'].count('y')

            # add epiref
            self.epiref_type = epiref_type
            # ----------------Handle sequence exceptions first---------------------


            if str(self.pd['seqfil']) == 'ge3d_elliptical':
               
                self = make_im3Dcs()

            #--------------------------Handle by apptype---------------------------

            if self.pd['apptype'] == 'im2D':

                self = make_im2D()
                self.is_kspace_complete = True

            elif self.pd['apptype'] == 'im2Dcs':

                self = make_im2Dcs()

            elif self.pd['apptype'] == 'im2Depi':

                self = make_im2Depi()
                self.is_kspace_complete = True

            elif self.pd['apptype'] == 'im2Depics':

                self = make_im2Depics()

            elif self.pd['apptype'] == 'im2Dfse':

                self = make_im2Dfse()
                self.is_kspace_complete = True

            elif self.pd['apptype'] == 'im2Dfsecs':

                self = make_im2Dfsecs()

            elif self.pd['apptype'] == 'im3D':

                self = make_im3D()
                self.is_kspace_complete = True

            elif self.pd['apptype'] == 'im3Dcs':

                self = make_im3Dcs()

            elif self.pd['apptype'] == 'im3Dute':

                self = make_im3Dute()
                self.is_kspace_complete = True

            else:
                raise(Exception('Could not find apptype.'))

        # ===================== Global modifications on Kspace ================

        #TODO if slices are in reversed order, flip them
        
        # in revamp make new axes, mainly for nifti io, and viewers
        # old : [rcvrs, phase, read, slice, time]
        # new : [read, phase, slice, time, rcvrs]

        # TODO moved these to the individual recons
        #self.data = np.moveaxis(self.data,[0,1,2,3,4],[4,1,0,2,3])
        # swap axes 0 and 1 so phase, readout etc is the final order
        #self.data = np.swapaxes(self.data,0,1)
        self.vdtype='kspace'
        self.to_local()

        if vj.config['default_space'] == None:
            pass
        elif vj.config['default_space'] == 'anatomical':
            self.to_anatomical()
        
        return self

    def to_imagespace(self, method='vnmrjpy'):
        """ Reconstruct MR images to real space from k-space.

        Generally this is done by fourier transform and corrections.
        Hardcoded for each 'seqfil' sequence.

        Args:
            method -- set either 'xrecon' or 'vnmrjpy'
        Updates attributes:
            data
            

        """
        # ========================== call Xrecon ==============================
        if method == 'xrecon':
        
            pass

        # =========================== Vnmrjpy custom fft ======================
        elif method == 'vnmrjpy':
            seqfil = str(self.pd['seqfil'])
            # this is for fftshift
            ro_dim = self.sdims.index('read')  # this should be default
            pe_dim = self.sdims.index('phase')  # this should be default
            pe2_dim = self.sdims.index('slice')  # slice dim is also pe2 dim
            
            sa = (ro_dim, pe_dim, pe2_dim)

            if seqfil in ['gems', 'fsems', 'mems', 'sems', 'mgems']:

                self.data = vj.core.recon._ifft(self.data,sa[0:2])

            elif seqfil in ['epip','epi']:

                self.data = vj.core.recon._ifft(self.data,sa[0:2])

            elif seqfil in ['ge3d','fsems3d','mge3d']:
                
                self.data = vj.core.recon._ifft(self.data,sa)

            elif seqfil in ['ge3d_elliptical']:

                self.data = vj.core.recon._ifft(self.data,sa)

            else:
                raise Exception('Sequence reconstruction not implemented yet')

        # wrapping up
        self.vdtype = 'imagespace'

        return self

