import vnmrjpy as vj
import numpy as np
from nipype.interfaces import fsl
import os
import time
import nibabel as nib
import copy
import shutil
import warnings

def COMPOSER(varr, varr_ref, workdir=None, keepfiles=False, log=True):
    """Phase match multiple receiver channels with help from a reference scan

    Args:
        varr (vj.varray) -- input sequence to phase match 
        varr_ref (vj.varray) -- short echo time reference scan
        workdir (boolean)  -- /path/to/temp/workdir
                                (if None, defaults to one in vj.config)
        keepfiles (boolean) -- set True to keep work files saved in [workdir]
        log (boolean) -- write log file 
    Return:
        varr (vj.varray) -- same as input varr, but with phase matched data

    Outline:
        Heavy use of FSL FLIRT and applyxfm via nipype. See ref for method

    Ref:
        Robinson et al.: Combining Phase Images from Array Coils Using a
        Short Echo Time Reference Scan (COMPOSER), (2017)
    """
    #sanity check
    if varr.vdtype == 'imagespace' and varr_ref.vdtype == 'imagespace':
        pass
    else:
        raise(Exception('Input data should be in image space'))
    # check reference echo time: is it short?
    et = float(varr_ref.pd['te'])*1000
    if  et > 2:
        print('Ref echo time is '+str(et)+' ms. Might bee too long.')

    rcvr_dim = int(vj.config['rcvr_dim'])
    time_dim = int(vj.config['et_dim'])

    if workdir == None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        workdir = vj.config['fsl_workdir']+'/composer_'+timestr
    # ------------------------FLIRT registration-------------------------------
    
    #ssos, save to nifti, register varr_ref to varr
    img_out = workdir+'/composer_img'
    ref_out = workdir+'/composer_ref'
    flirt_out = workdir+'/composer_flirtout'
    fullref_out = workdir+'/composer_fullimg'
    matrix_file = workdir+'/composer_invol2refvolmat'
    finref_out = workdir+'/composer_finrefout'
    composer_out = workdir+'/composer_finalout'
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
    flirt.inputs.interp = 'spline'
    res = flirt.run()
    # apply transform matrix to reference, but keep each channel
    applyxfm = fsl.preprocess.ApplyXFM()
    applyxfm.inputs.in_file = fullref_out+'.nii.gz'
    applyxfm.inputs.in_matrix_file = matrix_file
    applyxfm.inputs.out_file = finref_out+'.nii.gz'
    applyxfm.inputs.out_matrix_file = workdir+'/applyxfm_matrix'
    applyxfm.inputs.reference = img_out+'.nii.gz'
    applyxfm.inputs.interp = 'nearestneighbour'
    applyxfm.inputs.apply_xfm=True
    res = applyxfm.run()
    # load nifti data with nibabel
    data = nib.load(finref_out+'.nii.gz').get_fdata()
    # extract phase from aligned short echo time reference:
    ph_setr = data[:,:,:,4:]
    # align dimensions, so rcvr is dim4
    ph_setr = np.expand_dims(ph_setr,axis=time_dim)
    # make individual-channel magnitudes and phases 
    magn_idch = np.absolute(varr.data)
    ph_idch = np.arctan2(np.imag(varr.data),np.real(varr.data)) 
    #TODO this is for testing
    # save individual channels
    """
    varr_ph = copy.copy(varr)
    varr_setr = copy.copy(varr)
    varr_setr.data = ph_setr
    varr_ph.data = ph_idch
    ph_test_out = workdir+'/ph_test'
    ph_setr_out = workdir+'/ph_setr'
    vj.core.write_nifti(varr_ph,ph_test_out,\
                rcvr_to_timedim=True,combine_rcvrs=False)
    vj.core.write_nifti(varr_setr,ph_setr_out,\
                rcvr_to_timedim=True,combine_rcvrs=False)
    """
    # creating summed output
    csum = np.sum(magn_idch*np.exp(1j*(ph_idch-ph_setr)),axis=rcvr_dim)
    # final composer phase
    comp_ph = np.arctan2(np.imag(csum),np.real(csum))
    # magnitude weighted output
    comp_magn_w = np.sqrt(np.abs(np.sum(\
                magn_idch**2*np.exp(1j*(ph_idch-ph_setr)),axis=rcvr_dim)))
    comp_cmplx = comp_magn_w*np.exp(1j*comp_ph)
    # expand to rcvr_dim
    comp_cmplx = np.expand_dims(comp_cmplx,axis=-1)
    #update data
    varr.data = comp_cmplx
    if keepfiles==True:
        vj.core.write_nifti(varr,composer_out,\
            save_complex=True,rcvr_to_timedim=True,combine_rcvrs=False)
    elif keepfiles == False:
        shutil.rmtree(workdir)
    varr.description = 'phasematched'
    return varr

