import unittest
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt
import time
#import cupy as cp

RP={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False}
PLOTTING = False

class Test_hankelutils(unittest.TestCase):
    
    def test_lvl2_hankel_average(self):

        #rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
        #    'recontype':'k-t','fiber_shape':(4,128,21)}
        rp={'rcvrs':4,'filter_size':(21,21),'virtualcoilboost':False,\
            'recontype':'kx-ky','fiber_shape':(4,192,192)}
        stage = 0
        (p,q) = rp['filter_size']
        (rcvrs,m,n) = rp['fiber_shape']
        fiber = np.random.rand(*rp['fiber_shape'])
        
        hankel = vj.aloha.construct_hankel(fiber,rp)
        noise = np.random.rand(*hankel.shape)
        hankel_noisy = hankel+noise
        # average merthod old:
        start_old = time.time()
        fiber_avg = vj.aloha.deconstruct_hankel(hankel_noisy,stage,rp)
        hankel_avg1 = vj.aloha.construct_hankel(fiber_avg,rp)
        end_old = time.time()
        # new average method
        start_new = time.time()
        hankel_avg2 = vj.aloha.lvl2_hankel_average(hankel_noisy\
                                        ,rp['filter_size'],fiber.shape)
        end_new = time.time()
        print('old avg time: {}'.format(end_old-start_old))
        print('new avg time: {}'.format(end_new-start_new))
        plt.subplot(1,4,1)
        plt.imshow(np.real(hankel))
        plt.title('orig')
        plt.subplot(1,4,2)
        plt.imshow(np.real(hankel_avg1))
        plt.title('old avg')
        plt.subplot(1,4,3)
        plt.imshow(np.real(hankel_avg2))
        plt.title('new avg')
        plt.subplot(1,4,4)
        plt.imshow(np.real(hankel_avg2-hankel_avg1))
        plt.title('difference')
        plt.show()

    # this wont be implemented probably
    """
    def test_lvl2_hankel_weights(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,32,21)}
        filter_shape = (11,7)
        fiber_shape = (4,32,21)
        a = vj.aloha.lvl2_hankel_weights(filter_shape,fiber_shape)
    """
    # this works, but is really slow
    """
    def test_average_hankel(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,128,21),'stages':3}
        stage = 0
        hankel = np.random.rand(1770,308)
        start = time.time()
        hankel_avg = vj.aloha.average_hankel(hankel,stage,rp)
        print('hankel avg small : {}'.format(time.time()-start))

        rp={'rcvrs':4,'filter_size':(21,21),'virtualcoilboost':False,
            'recontype':'kx-ky','fiber_shape':(4,192,192),'stages':3}
        hankel = np.random.rand(29584,1764)
        start = time.time()
        hankel = vj.aloha.average_hankel(hankel,stage,rp)
        end = time.time()
        self.assertEqual(hankel.shape,(29584,1764))
        print('average big hankel time : {}'.format(end-start))
    """
    #--------------------------STANDARD TESTS----------------------------------
    def test_construct_hankel_2d(self):
        # this is the old one for time comparison
        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False}
        indata = np.random.rand(4,128,21)
        start = time.time()
        hankel = vj.aloha.construct_hankel_2d(indata,rp)
        end = time.time()
        self.assertEqual(hankel.shape,(1770,308))
        print('Construct_2d small hankel time : {}'.format(end-start))
        # test bigger kx ky
        rp={'rcvrs':4,'filter_size':(21,21),'virtualcoilboost':False}
        indata = np.random.rand(4,192,192)
        start = time.time()
        hankel = vj.aloha.construct_hankel_2d(indata,rp)
        end = time.time()
        self.assertEqual(hankel.shape,(29584,1764))
        print('Construct_2d big hankel time : {}'.format(end-start))

    def test_construct_hankel(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False}
        indata = np.random.rand(4,128,21)
        start = time.time()
        hankel = vj.aloha.construct_hankel(indata,rp)
        end = time.time()
        self.assertEqual(hankel.shape,(1770,308))
        print('Construct small hankel time : {}'.format(end-start))
        # test bigger kx ky
        rp={'rcvrs':4,'filter_size':(21,21),'virtualcoilboost':False}
        indata = np.random.rand(4,192,192)
        start = time.time()
        hankel = vj.aloha.construct_hankel(indata,rp)
        end = time.time()
        self.assertEqual(hankel.shape,(29584,1764))
        print('Construct big hankel time : {}'.format(end-start))

    def test_deconstruct_hankel(self):
        
        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,128,21)}
        hankel = np.random.rand(1770,308)
        stage = 0
        start = time.time()
        nd_data = vj.aloha.deconstruct_hankel(hankel, stage, rp)
        end = time.time()
        self.assertEqual(nd_data.shape,(4,128,21))
        print('Deconstruct small hankel time : {}'.format(end-start))

        rp={'rcvrs':4,'filter_size':(21,21),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,192,192)}
        hankel = np.random.rand(29584,1764)
        stage = 0
        start = time.time()
        nd_data = vj.aloha.deconstruct_hankel(hankel, stage, rp)
        end = time.time()
        self.assertEqual(nd_data.shape,(4,192,192))

        print('Deconstruct big hankel time : {}'.format(end-start))

    def test_make_kspace_weights(self):
        
        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,128,21),'stages':3}
        weights = vj.aloha.make_kspace_weights(rp)
        self.assertEqual(weights[1].shape,(4,64,21))

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':True,\
            'recontype':'kx-ky','fiber_shape':(4,128,128),'stages':3}
        weights = vj.aloha.make_kspace_weights(rp)
        self.assertEqual(weights[3].shape,(8,64,64))
        if PLOTTING == True:
            plt.imshow(np.absolute(weights[0][0,...]))
            plt.show()

    def test_init_kspace_stage(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'kx-ky','fiber_shape':(4,128,128),'stages':3}
        kspace = np.random.rand(4,128,128)
        stage = 1
        kspace_init = vj.aloha.init_kspace_stage(kspace,stage,rp)
        self.assertEqual(kspace_init.shape,(4,64,64))
    
        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':True,\
            'recontype':'k-t','fiber_shape':(4,128,21),'stages':3}
        kspace = np.random.rand(8,128,21)
        stage = 1
        kspace_init = vj.aloha.init_kspace_stage(kspace,stage,rp)
        self.assertEqual(kspace_init.shape,(8,64,21))

    def test_finalize_kspace_stage(self):

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'kx-ky','fiber_shape':(4,128,128),'stages':3}
        fullk = np.zeros((4,128,128))
        stagek = np.ones((4,64,64))
        kspace_full = vj.aloha.finish_kspace_stage(stagek,fullk,0,rp)
        self.assertEqual(kspace_full.shape,(4,128,128))
        self.assertEqual(kspace_full[1,64,64],0)
        self.assertEqual(kspace_full[1,60,60],1)

        rp={'rcvrs':4,'filter_size':(11,7),'virtualcoilboost':False,\
            'recontype':'k-t','fiber_shape':(4,128,21),'stages':3}
        fullk = np.zeros((4,128,21))
        stagek = np.ones((4,64,21))
        kspace_full = vj.aloha.finish_kspace_stage(stagek,fullk,0,rp)
        self.assertEqual(kspace_full.shape,(4,128,21))
        self.assertEqual(kspace_full[1,60,10],1)

