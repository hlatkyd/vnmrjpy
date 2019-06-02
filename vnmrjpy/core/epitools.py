import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
"""
Collection of helper functions for epi and epip k-space formation, ghost
correction, fieldmap correction and general preprocessing dealing with kspace.
"""
def _navigator_scan_correct(kspace, p, method='default'):
    """Correct epi kspace with reference images

    Corrects phases with the navigator scans after the readout reversal,
    but does not apply the additional triple reference scheme.
    The output kspace has the same dimensions as the input.
    Triple navigator echo schemes and others can be used thereafter.

    [1] Bruder et al: Image Reconstruction for Echo Planar Imaging with
    Nonequidistant k-Space Sampling, 1990, MRM       
    [2] F.Schmitt: Echo-Planar Imaging: Theory, Technique and Application,
    1998, Springer

    Args:
        kspace -- kspace, including reference scans. dimensions:

                    [rcvrs, time, slice, seg, phase, read]

        p -- procpar dictionary
        method -- specificator string, method specified here takes
                  priority over the one in config file
    Return:
        kspace -- navigator scan corrected kspace with same dims as input
    """
    # init
    if method == 'default':
        method = vj.config['epiref']
    if method == 'default':
        method = p['epiref_type']
    if method == 'none':
        return kspace
    # default vnmrj-like correction
    elif method == 'triple' or method == 'fulltriple':

        # check if data is eligible for this correction
        if (int(p['image'][0]) != 0 and
            int(p['image'][1]) != -2):
            raise(Exception('Navigator scans not found. Quitting.'))

        # phase correction of non-phase-encoded navgator scans
        ind = p['image'].index('0')
        stdrofilt = _phasefilter(kspace[:,ind:ind+1,...])
        stdromtffilt = _mtffilter(kspace[:,ind:ind+1,...])
        # phase ccorrection for reverse-readout navigator scan
        ind = p['image'].index('-2')
        revrofilt = _phasefilter(kspace[:,ind:ind+1,...])
        revromtffilt = _mtffilter(kspace[:,ind:ind+1,...])

        ind = [i for i, x in enumerate(p['image']) if x == '0']
        kspace_nav = kspace[:,ind,...]
        ind = [i for i, x in enumerate(p['image']) if x == '-2']
        kspace_revnav = kspace[:,ind,...]
        # 'normal' scans are labeled 'image' = 1
        ind = [i for i, x in enumerate(p['image']) if x == '1']
        kspace_img = kspace[:,ind,...]
        # reversed readout scans are labeled 'image' = -1
        ind = [i for i, x in enumerate(p['image']) if x == '-1']
        kspace_ref = kspace[:,ind,...]

        # creating final image filters
        stdfilt = _combined_filt(stdrofilt,stdromtffilt)
        revfilt = _combined_filt(revrofilt,revromtffilt)
        # correcting individual images
        kspace_img = _apply_filter(kspace_img,stdfilt) 
        kspace_ref = _apply_filter(kspace_ref,revfilt)
        kspace_nav = _apply_filter(kspace_nav,stdfilt)
        kspace_revnav = _apply_filter(kspace_revnav,revfilt)
       
        # concatenating volumes into the correct series

        ind = [i for i, x in enumerate(p['image']) if x == '-0']
        kspace[:,ind,...] = kspace_nav
        ind = [i for i, x in enumerate(p['image']) if x == '-2']
        kspace[:,ind,...] = kspace_revnav
        # 'normal' scans are labeled 'image' = 1
        ind = [i for i, x in enumerate(p['image']) if x == '1']
        kspace[:,ind,...] = kspace_img
        # reversed readout scans are labeled 'image' = -1
        ind = [i for i, x in enumerate(p['image']) if x == '-1']
        kspace[:,ind,...] = kspace_ref
        
        return kspace

    elif method == 'aloha':
        raise(Exception('ALOHA Not implemented yet'))
    else:
        raise(Exception('Incorrect method specification'))


def _navigator_echo_correct(kspace, npe, etl, method=None):
    """Basic navigator echo correction with various methods

    Perform navigator echo correction before the segments are merged, and after
    the readout lines are reversed. 

    Ref [1].: Kim et al.: Fast Interleaved Echo-Planar Imaging with Navigator:
    High Resolution anatomic and Functional Images at 4T, MRM, 1996
    ref [2].: O.Heid : Robust EPI Phase Correction, PROC ISMRM, 1997

    Args:
        kspace -- kspace in numpy.ndarray with dimensions of 
                        (rcvr,time,slices,nseg,npe,read)
        npe -- number of readout lines, including navigators
        etl -- echo train length
        method -- method specification string, if None it is figured out
    Return:
        kspace -- corrected kspace in the same deimnsions
    """ 
    # count navigator echos to determine method
    if type(method) == type(None):
        nnav = npe-etl-1
        if nnav == 0:
            method = 'none'
        elif nnav == 1:
            method = 'single'
        elif nnav == 2:
            method = 'dual'
        elif nnav == 3:
            method = 'triple'
        else: raise Exception('Unknown navigator correction scheme')

    elif method == 'default':
        method = vj.config['epinav']

    if method == 'none':
        return kpace    

    if method == 'single':
        # correct for magnitude offset between segments
        ro_proj = np.fft.fftshift(kspace, axes=-1)
        phase_filt = _phasefilter(kspace[...,:1,:])
        ro_proj = np.fft.ifft(ro_proj,axis=-1)
        magn = np.absolute(ro_proj)
        phase = np.arctan2(np.imag(ro_proj),np.real(ro_proj))
        # magnitude correction
        magn_corr_factor = np.mean(magn[...,:1,:],axis=3,\
                                keepdims=True)/magn[...,:1,:]
        ro_proj = ro_proj * magn_corr_factor
        # phase correction
        ro_proj = ro_proj * phase_filt
        kspace = np.fft.fft(ro_proj,axis=-1)
        kspace = np.fft.fftshift(kspace, axes=-1)
        return kspace
    #TODO
    elif method == 'dual':
        print('Warning, dual echo orr not implemented, doing nothing...')
        return kspace
        
    #TODO
    elif method == 'triple':
        
        nav1 = kspace[...,:1,:]
        nav3 = kspace[...,2:3,:]
        nav2_p = ( nav1 + nav3 ) / 2
        nav2_n = kspace[...,1:2,:]
        print('Warning, triple echo orr not implemented, doing nothing...')

        return kspace

def _remove_navigator_echos(kspace, etl):
    """Remove navigaotr and unused echos, return reduced kspace"""
    npe = kspace.shape[4]
    return kspace[...,(npe-etl):,:]

def _zerofill(kspace, phase_dim, nseg):
    """Zerofill kspace to fill the intended phase dimension"""
    shape = list(kspace.shape)
    shape[4] = phase_dim // nseg - shape[4]
    add_zeros = np.zeros((shape),dtype=kspace.dtype)
    return np.concatenate([kspace, add_zeros],axis=4)

#TODO
def _combine_segments_std(kspace, phase, order='linear'):
    """Combine segments in the same manner as vnmrj"""
    def _get_pe_order(nseg, phase, order='linear'):
        if order == 'linear':
            pass
        elif order == 'centric':
            pass
    pre_shape = list(kspace.shape)
    new_shape = pre_shape[[0,1,2,3,5]]
    new_shape[3] = phase
    kspace = np.reshape(kspace, new_shape)
    return kspace

def _arrange_pe(kspace, phase_order):
    """Rearrange PE dimension to combine segments to be in order"""
    kspace[:,:,:,np.array(phase_order),:] = copy.copy(kspace)
    return kspace

def _reverse_even(kspace, read_dim=4, phase_dim=3):
    """Reverse even echos (0,2,4...)"""
    kspace[...,0::2,:] = np.flip(kspace[...,0::2,:],axis=read_dim) 
    return kspace

def _reverse_odd(kspace, read_dim=4, phase_dim=3):
    """Reverse odd echos (1,3,5...)"""
    kspace[...,1::2,:] = np.flip(kspace[...,1::2,:],axis=read_dim) 
    return kspace

def _get_phaseorder_frompar(nseg, npe, etl, kzero):
    """Return phase reordering indices in a slice"""

    phase_order = []
    oneshot = [i*nseg for i in range(npe)]
    for i in range(nseg):
        phase_order += [j+i for j in oneshot ]
    
    return phase_order

def _get_navorder_frompar():

    pass

#TODO this is for compressed sensing or general>
def _get_phaseorder_frompetab(petab_file):
    """Read from file and return phase reordering indices in a slice"""
    pass

def triple_ref_correct(kspace, p):
    """Triple reference corection scheme on kspace

    Ref paper:
    [1] van der Zwaag et al: Minimization of Nyquist ghosting for
    Echo-Planar Imaging at Ultra-High fields based on a 'Negative Readout
    Gradient' Strategy, 2009, MRM

    Args:
        kspace -- navigator corrected kspace
        p -- procpar dictionary
    Return:
        kspace
    """
    ref_ind = [i for i,x in enumerate(p['image']) if x == '-1']
    kspace_ref = kspace[:,:,:,ref_ind,:]
    if len(ref_ind) != 1:
        raise(Exception('Triple ref can only handle 1 reference image'))
    img_ind = [i for i,x in enumerate(p['image']) if x == '1']
    kspace_img = kspace[:,:,:,img_ind,:]
    # just add negative RO scans to positive RO scans
    kspace = kspace_ref + kspace_img
    return kspace

#TODO
def fulltriple_ref_correct(kspace, p):
    """Full triple reference corection scheme on kspace

    Ref paper:
    [1] van der Zwaag et al: Minimization of Nyquist ghosting for
    Echo-Planar Imaging at Ultra-High fields based on a 'Negative Readout
    Gradient' Strategy, 2009, MRM

    Args:
        kspace -- navigator corrected kspace
        p -- procpar dictionary
    Return:
        kspace
    """
    ref_ind = [i for i,x in enumerate(p['image']) if x == '-1']
    kspace_ref = kspace[:,:,:,ref_ind,:]
    img_ind = [i for i,x in enumerate(p['image']) if x == '1']
    kspace_img = kspace[:,:,:,img_ind,:]
    # just add negative RO scans to positive RO scans
    kspace = kspace_ref + kspace_img
    return kspace

#------------------------------HELPERS-------------------------------------
def _phasefilter(nav):
    """Make phase filter from navigator scan"""
    nav = np.fft.fftshift(nav,axes=5)
    nav = np.fft.ifft(nav,axis=5)
    phase = np.arctan2(np.imag(nav),np.real(nav))
    filt =  np.exp(-1j * phase)
    return filt

def _mtffilter(nav, echo_average=True):
    """Return modulation transfer function filter for odd lines"""
    # phase correction
    nav = np.fft.fftshift(nav,axes=5)
    nav = np.fft.ifft(nav,axis=5)
    phase = np.arctan2(np.imag(nav),np.real(nav))
    phase_filt =  np.exp(-1j * phase)
    filt = nav * phase_filt
    if echo_average == True:
        #TODO check: is this viable?
        avgnum = 4  # average only the first 4
        # odd line correction with modulation transfer func
        even = np.mean(filt[...,0:avgnum*2:2,:],axis=4,keepdims=True)
        odd = np.mean(filt[...,1:avgnum*2+1:2,:],axis=4,keepdims=True)
        mtf = np.divide(even,odd,dtype='complex64')
        mtf[...,0::2,:] = 1  # even lines should not change
    if echo_average == False:
        # use only first 2 echo
        even = filt[...,0:1,:]
        odd = filt[...,1:2,:]
        mtf = np.divide(even,odd,dtype='complex64')
    return mtf

def _combined_filt(phasefilt, mtffilt):
    """Return final combined filter"""
    filt = phasefilt * mtffilt
    return filt

def _apply_filter(kspace, filt):
    """Apply filter to kspace

    Args:
        kspace (np.ndarray) -- raw preprocesed kspace, excluding reference scans
                shape : [rcvrs, time, slice, seg, phase, read]
        filt (np.ndarray) -- filter to be applied after FFT in RO dimension
                shape : [rcvrs, time, slice, seg, phase, read]
    Return:
        kspace
    """
    kspace = np.fft.fftshift(kspace,axes=5)
    kspace = np.fft.ifft(kspace,axis=5)
    kspace = kspace * filt
    kspace = np.fft.fft(kspace,axis=5) 
    kspace = np.fft.fftshift(kspace,axes=(5))
    return kspace 
