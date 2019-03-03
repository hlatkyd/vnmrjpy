import sys
import os
import numpy as np
import nibabel as nib
import vnmrjpy as vj
import warnings
import matplotlib.pyplot as plt

class KspaceMaker():
    """Class to build the k-space from the raw fid data based on procpar.

    Raw fid_data is numpy.ndarray(blocks, traces * np) format. Should be
    untangled based on 'seqcon' or 'seqfil' parameters.
    seqcon chars refer to (echo, slice, Pe1, Pe2, Pe3)

    Should support compressed sensing
    In case of Compressed sensing the reduced kspace is filled with zeros
    to reach the intended final shape

    Leave rest of reconstruction to other classes/functions

    INPUT:  fid data = np.ndarra([blocks, np*traces])
            fid header
            procpar

    METHODS:
            make(): 
                return kspace = nump.ndarray\
                            ([rcvrs, phase, read, slice, echo*time])

    """
    def __init__(self, fid_data, fidheader, procpar, verbose=False):
        """Reads procpar"""

        def _get_arrayed_AP(p):
            """check for arrayed pars in procpar

            Return: dictionary {par : array_length}
            """

            AP_dict = {}
            for par in ['tr', 'te', 'fa']:
            
                pass

            return AP_dict

        self.p = vj.io.ProcparReader(procpar).read()
        self.fid_header = fidheader
        self.rcvrs = str(self.p['rcvrs']).count('y')
        self.arrayed_AP = _get_arrayed_AP(self.p)
        apptype = self.p['apptype']
        self.config = vj.config
        self.verbose = verbose
        self.procpar = procpar
        # decoding skipint parameter
        # TODO
        # final kspace shape from config file
        self.dest_shape = (vj.config['rcvr_dim'],\
                            vj.config['pe_dim'],\
                            vj.config['ro_dim'],\
                            vj.config['slc_dim'],\
                            vj.config['et_dim'])
        self.pre_kspace = np.vectorize(complex)(fid_data[:,0::2],\
                                                fid_data[:,1::2])
        self.pre_kspace = np.array(self.pre_kspace,dtype='complex64')
        # check for arrayed parameters, save the length for later 
        self.array_length = vj.util.calc_array_length(fid_data.shape,procpar)
        self.blocks = fid_data.shape[0] // self.array_length
        if verbose:
            print('Making k-space for '+ str(apptype)+' '+str(self.p['seqfil'])+\
                ' seqcon: '+str(self.p['seqcon']))

    def print_fid_header(self):
        for item in self.fhdr.keys():
            print(str('{} : {}').format(item, self.fhdr[item]))

    def make(self):
        """Build k-space from fid data

        Return: 
            kspace=numpy.ndarray([rcvrs,phase,readout,slice,echo/time])
        """
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
    
            p = self.p
            rcvrs = int(p['rcvrs'].count('y'))
            (read, phase, slices) = (int(p['np'])//2, \
                                            int(p['nv']), \
                                            int(p['ns']))
            shiftaxis = (self.config['pe_dim'],self.config['ro_dim'])
            if 'ne' in p.keys():
                echo = int(p['ne'])
            else:
                echo = 1

            time = 1
            finalshape = (rcvrs, phase, read, slices,echo*time*self.array_length)
            final_kspace = np.zeros(finalshape,dtype='complex64')
           
            for i in range(self.array_length):
 
                kspace = self.pre_kspace[i*self.blocks:(i+1)*self.blocks,...]

                if p['seqcon'] == 'nccnn':
                    shape = (self.rcvrs, phase, slices, echo*time, read)
                    kspace = np.reshape(kspace, shape, order='C')
                    kspace = np.moveaxis(kspace, [0,1,4,2,3], self.dest_shape)

                elif p['seqcon'] == 'nscnn':

                    raise(Exception('not implemented'))

                elif p['seqcon'] == 'ncsnn':

                    preshape = (self.rcvrs, phase, slices*echo*time*read)
                    shape = (self.rcvrs, phase, slices, echo*time, read)
                    kspace = np.reshape(kspace, preshape, order='F')
                    kspace = np.reshape(kspace, shape, order='C')
                    kspace = np.moveaxis(kspace, [0,1,4,2,3], self.dest_shape)

                elif p['seqcon'] == 'ccsnn':

                    preshape = (self.rcvrs, phase, slices*echo*time*read)
                    shape = (self.rcvrs, phase, slices, echo*time, read)
                    kspace = np.reshape(kspace, preshape, order='F')
                    kspace = np.reshape(kspace, shape, order='C')
                    kspace = np.moveaxis(kspace, [0,1,4,2,3], self.dest_shape)
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
            self.kspace = final_kspace
            return final_kspace

        def make_im2Dcs():
            """
            These (*cs) are compressed sensing variants
            """

            def decode_skipint_2D(skipint):

               pass
 
            raise(Exception('not implemented'))

        def make_im2Depi():

            p = self.p
            kspace = self.pre_kspace
            print(kspace.shape)
            nseg = p['nseg']
            kzero = int(p['kzero'])  
            images = int(p['images'])  # repetitions
            time = images
            if p['navigator'] == 'y':
                pluspe = 1 + int(p['nnav'])  # navigator echo + unused
            else:
                pluspe = 1  # unused only
            
            print('images {}'.format(images))
            print('nseg {}'.format(nseg))
            print('ns {}'.format(p['ns']))
            
            if p['pro'] != 0:
                (read, phase, slices) = (int(p['nread']), \
                                            int(p['nphase']), \
                                            int(p['ns']))
            else:
                (read, phase, slices) = (int(p['nread'])//2, \
                                            int(p['nphase']), \
                                            int(p['ns']))
            
            if p['seqcon'] == 'ncnnn':

                preshape = (self.rcvrs, phase+pluspe, slices, time, read)
                print(kspace.size)
                tmp = np.zeros(preshape)
                print(tmp.size)
                kspace = np.reshape(kspace, preshape, order='c')

        def make_im2Depics():
            raise(Exception('not implemented'))
        def make_im2Dfse():
            warnings.warn('May not work correctly')
            kspace = self.pre_kspace
            p = self.p
            petab = vj.util.getpetab(self.procpar,is_procpar=True)
            nseg = int(p['nseg'])  # seqgments
            etl = int(p['etl'])  # echo train length
            kzero = int(p['kzero'])  
            images = int(p['images'])  # repetitions
            (read, phase, slices) = (int(p['np'])//2, \
                                            int(p['nv']), \
                                            int(p['ns']))
            shiftaxis = (self.config['pe_dim'],self.config['ro_dim'])

            echo = 1
            time = images

            phase_sort_order = np.reshape(np.array(petab),petab.size,order='C')
            # shift to positive
            phase_sort_order = phase_sort_order + phase_sort_order.size//2-1

            if p['seqcon'] == 'nccnn':

                #TODO check for images > 1
                preshape = (self.rcvrs, phase//etl, slices, echo*time, etl, read)
                shape = (self.rcvrs, echo*time, slices, phase, read)
                kspace = np.reshape(kspace, preshape, order='C')
                kspace = np.swapaxes(kspace,1,3)
                kspace = np.reshape(kspace, shape, order='C')
                # shape is [rcvrs, phase, slices, echo*time, read]
                kspace = np.swapaxes(kspace,1,3)
                kspace_fin = np.zeros_like(kspace)
                kspace_fin[:,phase_sort_order,:,:,:] = kspace
                kspace_fin = np.moveaxis(kspace_fin, [0,1,4,2,3], self.dest_shape)
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

            self.kspace = kspace
            return kspace

        def make_im2Dfsecs():
            raise(Exception('not implemented'))
        def make_im3D():

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
            
                preshape = (self.rcvrs,phase2,phase*echo*time*read)
                shape = (self.rcvrs,phase2,phase,echo*time,read)
                kspace = np.reshape(kspace,preshape,order='F')
                kspace = np.reshape(kspace,shape,order='C')
                kspace = np.moveaxis(kspace, [0,2,4,1,3], self.dest_shape)
                kspace = np.flip(kspace,axis=3)

            if p['seqcon'] == 'ncccn':
                preshape = (self.rcvrs,phase2,phase*echo*time*read)
                shape = (self.rcvrs,phase,phase2,echo*time,read)
                kspace = np.reshape(kspace,preshape,order='F')
                kspace = np.reshape(kspace,shape,order='C')
                kspace = np.moveaxis(kspace, [0,2,4,1,3], self.dest_shape)
    
            if p['seqcon'] == 'cccsn':
            
                preshape = (self.rcvrs,phase2,phase*echo*time*read)
                shape = (self.rcvrs,phase,phase2,echo*time,read)
                kspace = np.reshape(kspace,preshape,order='F')
                kspace = np.reshape(kspace,shape,order='C')
                kspace = np.moveaxis(kspace, [0,2,4,1,3], self.dest_shape)

            if p['seqcon'] == 'ccccn':
                
                shape = (self.rcvrs,phase2,phase,echo*time,read)
                kspace = np.reshape(kspace,shape,order='C')
                kspace = np.moveaxis(kspace, [0,2,4,1,3], self.dest_shape)

            self.kspace = kspace
            return kspace

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
        # ----------------Handle sequence exceptions first---------------------

        if self.verbose == True:
            print('KspaceMaker: making seqfil : {}'.format(self.p['seqfil']))


        if str(self.p['seqfil']) == 'ge3d_elliptical':
           
            kspace = make_im3Dcs()

        #--------------------------Handle by apptype---------------------------

        if self.p['apptype'] == 'im2D':

            kspace = make_im2D()

        elif self.p['apptype'] == 'im2Dcs':

            kspace = make_im2Dcs()

        elif self.p['apptype'] == 'im2Depi':

            kspace = make_im2Depi()

        elif self.p['apptype'] == 'im2Depics':

            kspace = make_im2Depics()

        elif self.p['apptype'] == 'im2Dfse':

            kspace = make_im2Dfse()

        elif self.p['apptype'] == 'im2Dfsecs':

            kspace = make_im2Dfsecs()

        elif self.p['apptype'] == 'im3D':

            kspace = make_im3D()

        elif self.p['apptype'] == 'im3Dcs':

            kspace = make_im3Dcs()

        elif self.p['apptype'] == 'im3Dute':

            kspace = make_im3Dute()

        else:
            raise(Exception('Could not find apptype. Maybe not implemented?'))

        # ---------------------Global modifications on Kspace------------------

        #TODO if slices are in reversed order, flip them

        return kspace

