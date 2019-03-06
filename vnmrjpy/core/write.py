import vnmrjpy as vj
import numpy as np
import nibabel as nib
import glob

def write_procapr(pd, out):
    """Write procpar file from procpar dictionary"""
    pass
def write_fdf(varray, out):
    """Write fdf files from varray data into .img direcotory specified in out"""
    pass

def write_nifti(varray,out, save_procpar=True, save_complex=False):
    """Write Nifti1 files from vnmrjpy.varray.
   
    Args:
        varray       -- vnmrjpy.varray object containing the data
        out          -- output path
        save_procpar -- (boolean) saves hidden procpar in the same directory
    """
    # check type of data
    if varray.data.shape < 3:
        raise(Exception('Data dimension error'))
    if is not varray.is_kspace_complete:
        raise(Exception('K-space is not completed'))



    #------------------ making the Nifti affine and header-----------------

    # main write
    if '.nii' in out:
        out_name = str(out)
    elif '.nii.gz' in out:
        out_name = str(out[:-3])
    else:
        out_name = str(out)+'.nii'
    header = _make_nifti_header(varray)
    affine = _make_nifti_affine(varray)

    data = varray.data

    img = nib.Nifti1Image(data, affine, header)
    nib.save(img,out_name)
    os.system('gzip -f '+str(out_name))
    vprint('write_nifti : '+str(out_name)+' saved ... ')

def _make_nifti_header(varray):
    """Make Nifti header from attributes of varray"""
    affine = _make_nifti_affine(varray.pd)
    header = nib.nifti1.Nifti1Header()
    data = varray.data
    header['xyzt_units'] = 2
    header['dim'][0] = len(self.data.shape)
    header['dim'][1:4] = data.shape[0:3]
    header['intent_name'] = varray.intent
    # TODO write procpar from pd maybe?
    header['aux_file'] = varray.procpar
    header.set_qform(affine, code=qform_code)

    return header

def _make_nifti_affine(varray):

    pass
