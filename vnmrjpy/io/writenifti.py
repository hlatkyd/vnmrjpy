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
                                    output_space='default',\
                                    input_space='local',\
                                    niftiheader=None):
        """Makes affine, and nifti header"""

        # ----------------------------INIT HELPER FUNCTIONS--------------------
        def _make_scanner_affine():
            """Make appropriate affine for Nifti header"""
            affine = vj.util.make_scanner_affine(self.procpar)
            return affine

        # ----------------------------MAIN INIT--------------------------------
        
        self.procpar = procpar
        self.verbose = verbose
        ppr = vj.io.ProcparReader(procpar)
        self.ppdict = ppr.read()
        self.output_space = output_space
        self.input_space = input_space

        # expand to timedim even if time is only 1
        if len(data.shape) == 3:
            self.data = np.expand_dims(data,axis=-1)
        else:
            self.data = data

        #------------------ making the Nifti affine and header-----------------
        
        # which coordinate system to write?
        if niftiheader==None:

            if output_space=='default':
                #default to config file
                output_space = vj.config['default_space']
            # this is the standard
            if input_space=='local' and output_space=='scanner':
                self.data = vj.util.to_scanner_space(self.data, self.procpar)
                self.affine = vj.util.make_scanner_affine(self.procpar)
            elif input_space=='scanner' and output_space=='scanner':
                self.affine = vj.util.make_scanner_affine(self.procpar)
            elif input_space=='local' and output_space=='rat_anatomical':
                self.data = vj.util.to_scanner_space(self.data, self.procpar)
                self.data = vj.util.scanner_to_rat_anatomical(self.data)
                self.affine = vj.util.make_rat_anatomical_affine(self.procpar)
            elif input_space=='scanner' and output_space=='rat_anatomical':
                self.data = vj.util.scanner_to_rat_anatomical(self.data)
                self.affine = vj.util.make_rat_anatomical_affine(self.procpar)
            elif input_space=='local' and output_space=='local':
                print('Warning! Writenifti : Falling back to cubic affine')
                self.affine = vj.util.make_cubic_affine(self.procpar)
            elif input_space=='rat_anatomical' and output_space=='rat_anatomical':
                self.affine = vj.util.make_rat_anatomical_affine(self.procpar)
            else:
                raise(Exception('Not implemented'))

            self.header = self._make_nifti_header()
        else:
            self.header = niftiheader
            self.affine = niftiheader.affine


    def write(self,out):
        """Saves 4D-7D nifti in .nii.gz format

        data = [dim0,dim1,dim2,timedim,receivers,etc...]

        Arguments:
            out -- save path
        """
        if '.nii' in out:
            out_name = str(out)
        elif '.nii.gz' in out:
            out_name = str(out[:-3])
        else:
            out_name = str(out)+'.nii'
        img = nib.Nifti1Image(self.data, self.affine, self.header)
        nib.save(img,out_name)
        os.system('gzip -f '+str(out_name))
        if self.verbose:
            print('WriteNifti : '+str(out_name)+' saved ... ')
    
    #TODO check if this is even needed
    '''
    def write6d(self,out):
        """Saves data in [x,y,z,t,rvr,real/imag] format to nifti

        Args:
            out -- save path
        """
        if len(self.data.shape) != 6:
            raise(Exception('Data not in 6D format'))
        if '.nii' in out:
            out_name = str(out)
        elif '.nii.gz' in out:
            out_name = str(out[:-3])
        else:
            out_name = str(out)+'.nii'
        img = nib.Nifti1Image(self.data, self.affine, self.header)
        nib.save(img,out_name)
        os.system('gzip -f '+str(out_name))
        if self.verbose:
            print('writeNifti : '+str(out_name)+' saved ... ')
    '''

    def _make_nifti_header(self):
        """Make Nifti header from scratch

        """
        #TODO make some distinction somehow
        if self.output_space == 'rat_anatomical':
            qform_code = 1
        elif self.output_space == 'scanner':
            qform_code = 1
        else:
            qform_code = 0
        p = self.ppdict
        swaparr, flipaxis, sliceaxis = vj.util.get_swap_array(p['orient'])
        header = nib.nifti1.Nifti1Header()
        # TODO leave matrix and use data dims?
        matrix_orig, dim_orig = vj.util.make_local_matrix(self.ppdict)
        matrix = vj.util.swapdim(swaparr,matrix_orig)

        header['xyzt_units'] = 2
        header['dim'][0] = len(self.data.shape)
        header['dim'][1] = matrix[0]
        header['dim'][2] = matrix[1]
        header['dim'][3] = matrix[2]
        header['dim'][4] = self.data.shape[3]
        header['intent_name'] = 'THEINTENT'
        header['aux_file'] = self.procpar
        header.set_qform(self.affine, code=qform_code)

        return header
