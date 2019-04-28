import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt

"""
Collection of helper functions for epi and epip k-space formation
and preprocessing.
"""

def epi_debug_plot(kspace, navigators, p):
    """Epi plotting utility to see reference scans and navigator echoes

    Kspace is assumed to be pre-corrected
    """
    PLOTCH = 1  # channel to plot
    PLOTSLC = kspace.shape[2]//2  # plot middle slice
    reftype = p['epiref_type']
    print('navigators shape {}'.format(navigators.shape))
    print('kspace shape in plt {}'.format(kspace.shape))
    # assume first volumes of kspace to be the references
    if reftype == 'triple' or reftype == 'fulltriple':
        ref1 = kspace[:,:,PLOTSLC,0,PLOTCH]
        ref2 = kspace[:,:,PLOTSLC,1,PLOTCH]
        ref3 = kspace[:,:,PLOTSLC,2,PLOTCH]
        img = kspace[:,:,PLOTSLC,3,PLOTCH]
        imgs = [ref1, ref2, ref3,img]
    elif reftype == 'single':
        ref1 = kspace[:,:,PLOTSLC,0,PLOTCH]
        refs = [ref1]

    testimg = np.ones_like(imgs[0])
    # fft the refs along readout
    fftrefs = []
    for i in [ref1, ref2]:
        #im = np.fft.fftshift(i,axes=1)
        im = i
        im = np.fft.ifft(im, axis=1)
        #im = np.fft.ifftshift(im, axes=1)
        #im = np.fft.ifft(im, axis=0)
        fftrefs.append(im)

    print('nav shape {}'.format(navigators.shape))
    navs = navigators[:,:,PLOTSLC,:,PLOTCH]

    rows = 4
    cols = len(imgs)
    # plot individual kspaces of refs
    fig = plt.figure(figsize=(15,10))
    for i in range(len(imgs)):
        # plotting navigators
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.absolute(navs[:,:,i]),aspect='auto')
        # plotting kspace images
        plt.subplot(rows,cols,i+1+cols)
        plt.imshow(np.absolute(imgs[i]),cmap='gray')
    
    # plotting fftd refs
    plt.subplot(rows,cols, 0+1+2*cols)
    plt.imshow(np.absolute(fftrefs[0]))
    plt.subplot(rows,cols, 1+1+2*cols)
    plt.imshow(np.absolute(fftrefs[1]))
    plt.subplot(rows,cols, 2+1+2*cols)
    plt.imshow(np.arctan2(np.imag(fftrefs[0]),np.real(fftrefs[0])))
    plt.subplot(rows,cols, 3+1+2*cols)
    plt.imshow(np.arctan2(np.imag(fftrefs[1]),np.real(fftrefs[1])))
    plt.show()

def refcorrect(kspace, p):
    """Phase correct epi kspace with reference images
    
    Ref paper: van der Zwaag et al: Minimization of Nyquist ghosting for
    Echo-Planar Imaging at Ultra-High fields based on a 'Negative Readout
    Gradient' Strategy, 2009, MRM

    Dimensions are in default space
    
    Args:
        kspace
        p
    """
    #TODO
    return kspace

def navcorrect(kspace, nav, p, method='default'):
    """Correct kspace phase with navigator echo

    Both navigator and kspace are in default space
    
    Args:
        kspace -- epi ksapce
        nav -- navigator echo kspace
        p -- procpar dictionary
        method -- navigator correction method, default is pointwise
    Return:
        kspace -- corrected kspace
    """
    if method == 'default':
        navcorr = vj.config['default']:
    else:
        raise(Exception('Not implemented method'))

    if navcorr == 'pointwise':

        pass
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
    """Remove unused and navigator echo from kspace data"""
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
