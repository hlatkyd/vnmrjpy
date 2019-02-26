import os
import glob
import numpy as np
import nibabel as nib
import math
import vnmrjpy as vj

class NiftiWriter():
    """Class to write Nifti1 files from procpar and image or kspace data.

    Dimensions, orientations in the input data and procpar must match!
    
    INPUT:
            procpar
            data = numpy.ndarray([phase, readout, slice, time])
            to_scanner (boolean) -- write nifti according to scanner coordinates if True
            niftiheader -- accepts premade header data
    METHODS:
            write(out)

                nifti output dimensions are [phase, readout, slice, time]
                nifti affine is created from the procpar data

    """
    def __init__(self, data, procpar, verbose=False,\
                                    to_scanner=None,\
                                    from_local=True,\
                                    niftiheader=None):
        """Makes affine, and nifti header"""

        # ----------------------------INIT HELPER FUNCTIONS--------------------
        def _make_scanner_affine():
            """Make appropriate affine for Nifti header"""
            affine = vj.util.make_scanner_affine(self.procpar)
            return affine

        def _make_local_matrix():

            if self.ppdict['apptype'] in ['im2Depi']:
                dread = float(self.ppdict['lro']) / float(self.ppdict['nread'])*2*10
                dphase = float(self.ppdict['lpe']) / float(self.ppdict['nphase'])*10
                dslice = float(self.ppdict['thk'])+float(self.ppdict['gap'])
                matrix = (int(self.ppdict['nphase']),int(self.ppdict['nread'])//2)
                dim = np.array([dphase,dread,dslice])
            if self.ppdict['apptype'] in ['im2Dfse','im2D']:
                dread = float(self.ppdict['lro']) / float(self.ppdict['np'])*2*10
                dphase = float(self.ppdict['lpe']) / float(self.ppdict['nv'])*10
                dslice = float(self.ppdict['thk'])+float(self.ppdict['gap'])
                matrix = (int(self.ppdict['nv']),int(self.ppdict['np'])//2)
                dim = np.array([dphase,dread,dslice])
            if self.ppdict['apptype'] in ['im3D','im3Dshim']:
                dread = float(self.ppdict['lro']) / float(self.ppdict['np'])*2*10
                dphase = float(self.ppdict['lpe']) / float(self.ppdict['nv'])*10
                dphase2 = float(self.ppdict['lpe2']) / float(self.ppdict['nv2'])*10
                matrix = (int(self.ppdict['nv']),\
                            int(self.ppdict['np'])//2,\
                            int(self.ppdict['nv2']))
                dim = np.array([dphase,dread,dphase2])

            return matrix


        def _make_scanner_header():
            """Make Nifti header from scratch"""

            p = self.ppdict
            swaparr, flipaxis = vj.util.get_swap_array(p['orient'])
            header = nib.nifti1.Nifti1Header()
            matrix = _make_local_matrix()
            
            #
            dim_info
            if self.ppdict['apptype'] in ['im3D','im3Dshim']:
                header.set_data_shape(matrix)
                header.set_dim_info(phase=1,freq=0,slice=2)
                header.set_xyzt_units(xyz='mm')
                header.set_qform(aff, code='scanner')

            elif self.ppdict['apptype'] in ['im2D']:
                header.set_data_shape(matrix)
                header.set_dim_info(phase=0,freq=1,slice=2)
                header.set_xyzt_units(xyz='mm')
                header.set_qform(aff, code='scanner')

            elif self.ppdict['apptype'] in ['im2Dfse','im2Depi']:
                header.set_data_shape(matrix)
                header.set_dim_info(slice=2,phase=1,freq=0)
                header.set_xyzt_units(xyz='mm')
                header.set_qform(aff, code='scanner')
                if self.ppdict['apptype'] == 'im2Depi':
                    header.set_slice_duration(float(self.ppdict['slicetr']))

            return header


        # ----------------------------MAIN INIT--------------------------------

        self.verbose = verbose
        ppr = vj.io.ProcparReader(procpar)
        self.ppdict = ppr.read()

        if len(data.shape) == 3:
            self.data = np.expand_dims(data,axis=-1)
        elif len(data.shape) == 4:
            self.data = data
        else:
            print('datashape:'+str(data.shape))
            print('Wrong shape of input data')
            return

        #------------------ making the Nifti affine and header-----------------
        
        # which coordinate system to write?
        if niftiheader==None:
            if to_scanner==None:
            #default to config file
                self.coordinate_system = vj.config['default_space']
            elif to_scanner==True:
                self.coordinate_system = 'scanner'
            else:
                raise(Exception('Not implemented yet'))
            # this is the standard
            if from_local==True and self.coordinate_system=='scanner':
                self.data = vj.util.to_scanner_space(self.data, self.procpar)
                self.affine = vj.util.make_scanner_affine(self.procpar)
                self.header = _make_scanner_header()
            else:
                raise(Exception('not implemented'))
        else:
            self.header = niftiheader
            self.affine = niftiheader.affine


    def write(self,out):
        """Saves nifti in .nii.gz format

        Arguments:
            out -- save path
        """
        if '.nii' in out:
            out_name = str(out)
        elif '.nii.gz' in out:
            out_name = str(out[:-3])
        else:
            out_name = str(out)+'.nii'
        img = nib.Nifti1Image(self.data, self.affine, self.hdr)
        #img.update_header()
        nib.save(img,out_name)
        os.system('gzip -f '+str(out_name))
        if self.verbose:
            print('writeNifti : '+str(out_name)+' saved ... ')
                
