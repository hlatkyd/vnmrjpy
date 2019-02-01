import unittest
import numpy as np
import vnmrjpy as vj
import time
import matplotlib.pyplot as plt
import copy

# check execution time
TEST_PERFORMANCE=True
ITER=100

def plot_test(lst):

    n = len(lst)

    for num, item in enumerate(lst):

        plt.subplot(1,n,num+1)
        plt.imshow(np.absolute(item))
        #name = [k for k,v in locals().iteritems() if v==item][0]
        plt.title(str(num))
    plt.show()

class Test_jitutils(unittest.TestCase):

    def test_avg_hankel_lvl1(self):

        pass

    def test_avg_hankel_lvl2(self):

        rp = {'fiber_shape':(4,192,192),'filter_size':(21,21),'recontype':\
                'kx-ky'}

        a = np.random.rand(*rp['fiber_shape'])
        b = np.random.rand(*rp['fiber_shape'])
        stage = 0
        fiber = np.vectorize(complex)(a,b)
        fiber = np.array(fiber,dtype='complex64')
        filter_size = rp['filter_size']
        block_shape = (fiber.shape[1]-filter_size[0]+1,filter_size[0])
        rcvrs = fiber.shape[0]
        # constructing noisy hankel
        hankel = vj.aloha.construct_lvl2_hankel(fiber,filter_size)
        noise = np.random.rand(*hankel.shape)
        hankel_noisy = hankel + noise*2
        # new averaging
        if TEST_PERFORMANCE:
            start = time.time()
            for _ in range(ITER):
                avg = vj.aloha.avg_lvl2_hankel(hankel_noisy,\
                                            block_shape,rcvrs)

            end = time.time()
            print('averaging time for hankel {}'.format((end-start)/ITER))
        else:
            avg = vj.aloha.avg_lvl2_hankel(copy.copy(hankel_noisy),\
                                            block_shape,rcvrs)
        # original averaging
        fiber_avg = vj.aloha.deconstruct_hankel(copy.copy(hankel_noisy),stage,rp)
        avg_orig = vj.aloha.construct_hankel(fiber_avg,rp)
        # difference
        diff = avg - avg_orig
        # plotting
        self.assertEqual(True,np.allclose(\
                            diff,np.zeros_like(diff), atol=1e-5))
        plot_test([hankel,hankel_noisy,avg, avg_orig, diff]) 

    def test_construct_hankel(self):
    
        rp = {'fiber_shape':(4,192,192),'filter_size':(21,21),'recontype':\
                'kx-ky'}
        stage = 0
        recontype = rp['recontype']
        recontype_int = 1  # if kx ky
        fiber_shape = rp['fiber_shape']
        a = np.random.rand(*rp['fiber_shape'])
        b = np.random.rand(*rp['fiber_shape'])
        fiber = np.vectorize(complex)(a,b)
        fiber = np.array(fiber,dtype='complex64')
        filter_size = rp['filter_size']
        hankel = vj.aloha.construct_lvl2_hankel(fiber,filter_size)
        #for _ in range(10):
        #    start = time.time()
        #    hankel = vj.aloha.construct_lvl2_hankel(fiber,filter_size)
        #    print(time.time()-start)

