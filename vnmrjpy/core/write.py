import vnmrjpy as vj
from vnmrjpy.core.utils import vprint
import numpy as np
import nibabel as nib
import glob
from shutil import copyfile
import warnings
import os

def write_fdf(varray, out):
    """Write fdf files from varray data into .img direcotory specified in out"""
    pass

def write_nifti(varr,out, save_procpar=True,\
                        save_complex=False,\
                        rcvr_to_timedim=False,\
                        complex_pairs='magn-phase',\
                        combine_rcvrs='ssos',\
                        cut_to_3d=False):
    """Write Nifti1 files from vnmrjpy.varray.
  
    You can specify the data to be saved 
 
    Args:
        varray       -- vnmrjpy.varray object containing the data
        out          -- output path
        save_procpar -- (boolean) saves procpar json in the same directory
        save_complex -- save complex data in real-imag pairs along a dimension
        only4dim     -- compress data into 4 dimensions , if possible
        
    """
    # sanity check for input options

    if combine_rcvrs !=None and save_complex == True:
        #TODO
        pass

    # check type of data
    if len(varr.data.shape) < 3:
        raise(Exception('Data dimension error'))
    if not varr.is_kspace_complete:
        raise(Exception('K-space is not completed'))
    #------------------ making the Nifti affine and header-----------------

    # main write
    if '.nii' in out:
        out_name = str(out)
    elif '.nii.gz' in out:
        out_name = str(out[:-3])
    else:
        out_name = str(out)+'.nii'
  
    data = varr.data 
    
    # OPTION : save_complex----------------------------------------------------

    # to save complex data, real and imaginary parts are added
    # to receiver channel
    if save_complex:
        rch = 4

        if complex_pairs =='real-imag':
            data = np.concatenate([np.real(data), np.imag(data)],axis=rch)
        elif complex_pairs == 'magn-phase':
            magn = np.absolute(data)
            phase = np.arctan2(np.imag(data),np.real(data))
            data = np.concatenate([magn,phase],axis=rch)
        else:
            raise(Exception('Wrong complex_pairs'))

    # OPTION : rcvr_to_timedim-------------------------------------------------

    if rcvr_to_timedim:
        rch = 4
        timedim = data.shape[rch]*data.shape[3]
        newshape = (data.shape[0],data.shape[1],data.shape[2],timedim)
        data = np.reshape(data,newshape, order='c')

    # OPTION : combine_rcvrs---------------------------------------------------

    if combine_rcvrs == 'ssos':

        data = vj.core.recon.ssos(data,axis=4)
        # cut dimensions more than 3. This is mainly for use with certain FSL calls
        if cut_to_3d:
            if len(data.shape) == 4:
                data = data[...,0]
            elif len(data.shape) == 5:
                data = data[...,0,0]
            else:
                raise(Exception('Not implemented'))
    # set affine as none to use the one in header
    img = nib.Nifti1Image(data, None, varr.nifti_header)
    # create directories if not exists
    basedir = out_name.rsplit('/',1)[0]
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    nib.save(img,out_name)
    os.system('gzip -f '+str(out_name))
    vprint('write_nifti : '+str(out_name)+' saved ... ')

    if save_procpar:
        new_pp = out_name[:-3]+'procpar'
        try:
            copyfile(varr.procpar, new_pp)
        except:
            pdname = out_name[:-3]+'json'
            vj.core.utils.savepd(varr.pd,pdname)
            # TODO consider saving only in json
            #warnings.warn('Could not copy procpar. Saved dictionary as json.')

