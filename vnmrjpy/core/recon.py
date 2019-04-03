import vnmrjpy as vj
import numpy as np

"""
Collection of basic functions for reconstruction.

"""

def _ifft(data, dims):
    """Take the inverse fourier transform of kspace data"""
    data = np.fft.fftshift(data,axes=dims) 
    if len(dims) == 2:
        data = np.fft.ifft2(data,axes=dims,norm='ortho')
    elif len(dims) == 3:
        data = np.fft.ifftn(data,axes=dims,norm='ortho')
    data = np.fft.ifftshift(data,axes=dims)
    return data

def _fft(data, dims):
    """Take the inverse fourier transform of imagespace data"""
    pass

def ssos(data,axis=4):
    """Return squared sum of squares combination of receiver data"""
    # if 4 dim, just return the absolute
    if len(data.shape) == 4:
        return np.absolute(data)
    else:
        data = np.sqrt(np.sum(np.absolute(data)**2,axis=axis))
        return np.array(data,dtype='float32')
    
