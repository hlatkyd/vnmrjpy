import vnmrjpy as vj
import numpy as np
from nipype.interfaces import fsl
import os
import time
import nibabel as nib

def COMPOSER(varr, varr_ref, workdir=None, keepfiles=False):
    """Phase match multiple receiver channels with help from a reference scan

    Args:
        varr (vj.varray) -- input sequence to phase match 
        varr_ref (vj.varray) -- short echo time reference scan
        workdir (boolean)  -- /path/to/temp/workdir (default in config)
        keepfiles (boolean) -- set True to keep work files saved in [workdir]
    Return:
        varr_match (vj.varray)

    Outline:
        

    Ref:
        COMPOSER paper
    """

    if workdir == None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        workdir = vj.config['fsl_workdir']+'/'+timestr
    # ------------------------FLIRT registration-------------------------------
    
    print('input shape {}'.format(varr.data.shape))
    print('input tpye {}'.format(varr.data.dtype))
    #ssos, save to nifti, register varr_ref to varr
    img_out = workdir+'/composer_img'
    ref_out = workdir+'/composer_ref'
    flirt_out = workdir+'/composer_flirtout'
    fullref_out = workdir+'/composer_fullimg'
    matrix_file = workdir+'/composer_invol2refvolmat'
    finref_out = workdir+'/composer_finrefout'
    composer_out = workdir+'/test_finalout'
    # flirt cannot handle more than 3D, so cut data
    
    vj.core.write_nifti(varr,img_out,cut_to_3d=True)
    vj.core.write_nifti(varr_ref,ref_out, cut_to_3d=True)
    vj.core.write_nifti(varr_ref,fullref_out,\
                save_complex=True,rcvr_to_timedim=True,\
                combine_rcvrs=False)
    #register short echo time reference to main image
    flirt = fsl.FLIRT()
    flirt.inputs.in_file = ref_out+'.nii.gz'
    flirt.inputs.reference = img_out+'.nii.gz'
    flirt.inputs.out_file = flirt_out+'.nii.gz'
    flirt.inputs.out_matrix_file = matrix_file
    print(flirt.cmdline)
    res = flirt.run()
    # apply transform matrix to reference, but keep each channel
    applyxfm = fsl.preprocess.ApplyXFM()
    applyxfm.inputs.in_file = fullref_out+'.nii.gz'
    applyxfm.inputs.in_matrix_file = matrix_file
    applyxfm.inputs.out_file = finref_out+'.nii.gz'
    applyxfm.inputs.reference = img_out+'.nii.gz'
    applyxfm.inputs.apply_xfm=True
    res = applyxfm.run()
    # load nifti data with nibabel
    data = nib.load(finref_out+'.nii.gz').get_fdata()
    # extract phase from aligned short echo time reference:
    ph_setr = data[:,:,:,4:]
    # align dimensions, so rcvr is dim4
    ph_setr = np.expand_dims(ph_setr,axis=3)
    print('ph_setr shape {}'.format(ph_setr.shape))
    # make individual-channel magnitudes and phases 
    magn_idch = np.absolute(varr.data)
    ph_idch = np.arctan2(np.imag(varr.data),np.real(varr.data)) 
    # creating summed output
    csum = np.sum(magn_idch*np.exp(1j*(ph_idch-ph_setr)),axis=4)
    # final composer phase
    comp_ph = np.arctan2(np.imag(csum),np.real(csum))
    # magnitude weighted complex output
    comp_magn_w = np.sqrt(np.abs(np.sum(\
                magn_idch**2*np.exp(1j*(ph_idch-ph_setr)),axis=4)))
    # expand to rcvr_dim
    comp_magn_w = np.expand_dims(comp_magn_w,axis=-1)
    #update data
    varr.data = comp_magn_w
    print(varr.data.shape)
    vj.core.write_nifti(varr,composer_out,\
            save_complex=True,rcvr_to_timedim=True,combine_rcvrs=False)
    

