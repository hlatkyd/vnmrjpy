import unittest
import numpy as np
import vnmrjpy as vj
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

class Test_fftconvolve(unittest.TestCase):

    def test_fftconvolve(self):

        hankel = np.random.rand(50,20)
        kernel = np.fliplr(np.eye(hankel.shape[1]*2-1))
        #kernel = np.ones((hankel.shape[1],hankel.shape[1]))
        (m,n) = hankel.shape
        div = np.array([ [min(n,j+i+1) for j in range(n)] for i in range(m) ])
        div = np.minimum(div,np.flipud(np.fliplr(div)))
        hankel_new = convolve(hankel,kernel,mode='constant',cval=0)
        hankel_new_fft = fftconvolve(hankel,kernel,mode='same')
        hankel_new_fft_own = vj.aloha.fftconvolve(hankel,kernel,mode='same')
        print('hankel new shape {}'.format(hankel_new.shape))

        hankel_new = hankel_new / div
        hankel_new_fft = hankel_new_fft / div
