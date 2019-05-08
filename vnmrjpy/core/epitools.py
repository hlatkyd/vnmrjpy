import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt
import copy
"""
Collection of helper functions for epi and epip k-space formation
and preprocessing.
"""

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

def fulltriple_ref_correct(kspace, p):
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
    img_ind = [i for i,x in enumerate(p['image']) if x == '1']
    kspace_img = kspace[:,:,:,img_ind,:]
    # just add negative RO scans to positive RO scans
    kspace = kspace_ref + kspace_img
    return kspace

def fliprevro(kspace, p):
    """Flip kspace data where readout direction was reversed"""

    ind = [i for i, x in enumerate(p['image']) if x in ['-2','-1']]
    kspace[:,:,:,ind,:] = np.flip(kspace[:,:,:,ind,:],axis=0)
    return kspace

def _phasefilter(nav4d, roaxis=0):
    """Return phase corrected navigator scan"""
    nav4d = np.fft.fftshift(nav4d,axes=roaxis)
    nav4d = np.fft.ifft(nav4d,axis=roaxis)
    phase = np.arctan2(np.imag(nav4d),np.real(nav4d))
    filt =  np.exp(-1j * phase)
    return filt

def _mtffilter(nav4d, roaxis=0, echo_average=True):
    """Return modulation transfer function filter for odd lines"""

    if roaxis !=0:
        raise(Exception('roaxis !=0 wont work for now....'))
    # phase correction
    nav4d = np.fft.fftshift(nav4d,axes=roaxis)
    nav4d = np.fft.ifft(nav4d,axis=roaxis)
    phase = np.arctan2(np.imag(nav4d),np.real(nav4d))
    phase_filt =  np.exp(-1j * phase)
    filt = nav4d * phase_filt
    
    if echo_average == True:
        # odd line correction with modulation transfer func
        even = np.mean(filt[:,0::2,:,:],axis=1)
        odd = np.mean(filt[:,1::2,:,:],axis=1)
        mtf = np.divide(even,odd,dtype='complex64')
        mtf = np.expand_dims(mtf,axis=1)
        mtf = np.repeat(mtf,nav4d.shape[1],axis=1)
        mtf[:,0::2,:,:] = 1  # even lines should not change

    if echo_average == False:
        raise(Exception('not implemented yet'))

    return mtf

def _combined_filt(phasefilt, mtffilt, target_shape):
    """Return final combined filter"""
    filt = phasefilt * mtffilt
    slices = target_shape[3] 
    filt = np.expand_dims(filt,axis=3)
    filt = np.repeat(filt,slices,axis=3)
    return filt

def _apply_filter(kspace, filt):
    """Apply filter to kspace

    Args:
        kspace (np.ndarray) -- raw preprocesed kspace, excluding reference scans
        filt (np.ndarray) -- filter to be applied after FFT in RO dimension
    Return:
        kspace
    """
    kspace_before = copy.copy(kspace)
    kspace = np.fft.fftshift(kspace,axes=0)
    kspace = np.fft.ifft(kspace,axis=0)
    kspace = kspace * filt
    kspace = np.fft.fft(kspace,axis=0) 
    kspace = np.fft.fftshift(kspace,axes=(0,1))
    return kspace 

def navscan_correct(kspace, p, method='default'):
    """Phase correct epi kspace with reference images

    [1] Bruder et al: Image Reconstruction for Echo Planar Imaging with
    Nonequidistant k-Space Sampling, 1990, MRM       
    [2] F.Schmitt: Echo-Planar Imaging: Theory, Technique and Application,
    1998, Springer

    Dimension layout is in vnmrjpy default space: (read,phase,slice,time,rcvr)
    
    Args:
        kspace -- kspace, including reference scans
        p -- procpar dictionary
        method -- specificator string, method specified here takes
                priority over the one in config file
    Return:
        kspace -- navigator scan corrected kspace 
    """
    # init
    if method == 'default' and vj.config['epiref'] == 'default':
        method = p['epiref_type']
    elif method == 'default':
        pass
    else:
        method = vj.config['epiref']
    
    if method == 'none':
        pass

    # default vnmrj-like correction
    elif method == 'triple' or method == 'fulltriple':

        # check if data is eligible for this correction
        if (int(p['image'][0]) != 0 and
            int(p['image'][1]) != -2):
            raise(Exception('Navigator scans not found. Quitting.'))

        # phase correction of non-phase-encoded navgator scans
        ind = p['image'].index('0')
        stdrofilt = _phasefilter(kspace[:,:,:,ind,:])
        stdromtffilt = _mtffilter(kspace[:,:,:,ind,:])

        ind = p['image'].index('-2')
        revrofilt = _phasefilter(kspace[:,:,:,ind,:])
        revromtffilt = _mtffilter(kspace[:,:,:,ind,:])

        ind = [i for i, x in enumerate(p['image']) if x == '-0']
        kspace_nav = kspace[:,:,:,ind,:]
        ind = [i for i, x in enumerate(p['image']) if x == '-2']
        kspace_revnav = kspace[:,:,:,ind,:]
        # 'normal' scans are labeled 'image' = 1
        ind = [i for i, x in enumerate(p['image']) if x == '1']
        kspace_img = kspace[:,:,:,ind,:]
        # reversed readout scans are labeled 'image' = -1
        ind = [i for i, x in enumerate(p['image']) if x == '-1']
        kspace_ref = kspace[:,:,:,ind,:]

        # creating final image filters
        stdfilt = _combined_filt(stdrofilt,stdromtffilt,kspace_img.shape)
        revfilt = _combined_filt(revrofilt,revromtffilt,kspace_ref.shape)
        # correcting individual images
        
        kspace_img = _apply_filter(kspace_img,stdfilt) 
        kspace_ref = _apply_filter(kspace_ref,revfilt)
        kspace_nav = _apply_filter(kspace_nav,stdfilt)
        kspace_revnav = _apply_filter(kspace_revnav,revfilt)
       
        # concatenating volumes into the correct series

        ind = [i for i, x in enumerate(p['image']) if x == '-0']
        kspace[:,:,:,ind,:] = kspace_nav
        ind = [i for i, x in enumerate(p['image']) if x == '-2']
        kspace[:,:,:,ind,:] = kspace_revnav
        # 'normal' scans are labeled 'image' = 1
        ind = [i for i, x in enumerate(p['image']) if x == '1']
        kspace[:,:,:,ind,:] = kspace_img
        # reversed readout scans are labeled 'image' = -1
        ind = [i for i, x in enumerate(p['image']) if x == '-1']
        kspace[:,:,:,ind,:] = kspace_ref
        
    elif method == 'aloha':
        raise(Exception('Not implemented'))
    else:
        raise(Exception('Incorrect method specification'))

    return kspace

def navecho_correct(kspace, nav, p, method='default',timeavg=False):
    """Correct kspace phase with navigator echo

    Both navigator and kspace are in default space
    
    Args:
        kspace -- epi ksapce
        nav -- navigator echo kspace
        p -- procpar dictionary
        method -- navigator correction method, default is pointwise
        timeavg (boolean) -- If True, an average navigator is created to boost
                            navigator SNR
    Return:
        kspace -- corrected kspace
    """
    # get slice dimension


    if method == 'default':
        navcorr = vj.config['epinav']
    else:
        raise(Exception('Not implemented method'))

    if navcorr == 'pointwise':

        if timeavg == True:

            for slc in range(kspace.shape[2]):
                pass

        else:
            #re = np.real()
            pass
            navpshase = np.arctan2(np.imag(nav),np.real(nav))
                
    # do for each receiver individually
    #for i in range(kspace.shape[])
    return kspace

def _get_navigator_echo(kspace,pd,phase_dim=1):
    """Return navigator echo from full received data"""

    try:
        navs = int(pd['nnav'])
    except:
        pass
    if pd['navigator'] == 'y':
        if navs == 1:
            navigator = kspace[:,0:1,:,:,:]
        else:
            raise(Exception('more than one navigators not implemented yet'))
        return navigator
    else:
        return None

def _prepare_shape(kspace,pd,phase_dim=1):
    """Remove unused and navigator echo from kspace data
    
    By default, the first echos are navigators, then 1 unused line follows.

    More navigators at the start or end, are not supported, but would be wise
    to implement sometime

    Args:
        kspace -- unprepared kspace with unused echo and navigator echos
        pd -- procpar dictionary
    Return:
        kspace -- kspace with echos stripped
    """
    navind = _get_navigator_echo_index(pd)

    nnav = int(pd['nnav'])
    if nnav > 1:
        raise(Exception('More than 1 navigators are not implemented'))
    if pd['navigator'] == 'y':
        kspace = kspace[:,2:,:,:,:]
    else:
        kspace = kspace[:,1:,:,:,:]
    return kspace

def _kzero_shift(kspace,p,phase_dim=1):
    """Shift data so first echo is at the proper space according to kzero"""

    kzero = int(p['kzero'])
    kspace = np.roll(kspace,kzero,axis=phase_dim)
    return kspace

def _reverse_even(kspace, phase_dim=1):
    """Reverse even echoes"""
    odd = kspace[:,0::2,...]
    oddflip = np.flip(odd,axis=2)
    kspace[:,0::2,...] = oddflip
    return kspace

def _reverse_odd():
    pass

def _get_navigator_echo_index(p):
    """Return navigator echo positions along PE axis"""
    
    # number of navigators
    nnva = int(p['nnav'])
    nseg = int(p['nseg'])
    etl = int(p['etl'])

    pass

