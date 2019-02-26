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

        def _make_scanner_header():
            """Make Nifti header from scratch"""

            p = self.ppdict
            swaparr, flipaxis, sliceaxis = vj.util.get_swap_array(p['orient'])
            header = nib.nifti1.Nifti1Header()
            matrix_orig, dim_orig = vj.util.make_local_matrix(self.ppdict)
            matrix = vj.util.swapdim(swaparr,matrix_orig)
            print(matrix)
            
            if self.ppdict['apptype'] in ['im3D','im3Dshim']:
                header.set_data_shape(matrix)
                header.set_xyzt_units(xyz='mm')
                header.set_qform(aff, code='scanner')

            elif self.ppdict['apptype'] in ['im2D','im2Dfse','im2Dcs']:
                #header.set_xyzt_units(xyz='mm')
                header.set_qform(self.affine, code=1)
                #header['qform_code'] = 1
        
                header['xyzt_units'] = 2
                header['dim'][0] = 4
                header['dim'][1] = matrix[0]
                header['dim'][2] = matrix[1]
                header['dim'][3] = matrix[2]
                header['dim'][4] = 1
                header['intent_name'] = 'THEINTENT'
                header['aux_file'] = 'THEAUXILIARYFILE'
                print(header['dim'])
            else:
                raise(Exception('not implemented'))
    
            return header


        # ----------------------------MAIN INIT--------------------------------
        
        self.procpar = procpar
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
                print('niftiwriter HERE')
                print(self.data.shape)
                self.data = vj.util.to_scanner_space(self.data, self.procpar)
                # check for X gradient reversion
                #self.data = vj.util.corr_x_flip(self.data)
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
        print(self.data.shape)
        img = nib.Nifti1Image(self.data, self.affine, self.header)
        print('final header')
        print(img.header['dim'])
        #img.update_header()
        nib.save(img,out_name)
        os.system('gzip -f '+str(out_name))
        if self.verbose:
            print('writeNifti : '+str(out_name)+' saved ... ')
                
