import numpy as np
import nibabel as nib
from scipy.signal import gaussian
import vnmrjpy as vj

class SkipintDecoder():
    """Handles decoding of varios skipint formats

    skipint is the parameter responsible for choosing which k-space lines are
    acquired during compressed sensing acquisition

    """

    def __init__(self,procpar_dict):
        """Gather parameters from procpar for decoding
        
        Args:
            procpar_dict (dict) -- procpar dictionary, created by ProcparReader 
        """
        #check skiptab
        if 'skiptab' in procpar_dict.keys() and \
            procpar_dict['skiptab'] == 'y':
            pass
        else:
            raise(Exception("skiptab doesn't exist or set to 'n'."))
        self.skipint = procpar_dict['skipint']
        self.p = procpar_dict
        

    def make_skiptensor(self):
        """Main routine, makes a mask of the acquired lines

        Return:
            skiptensor (np.ndarray) -- multidimensional array, corresponding
                                    to a fiber in kspace. values are 1 if
                                    the kspace line is acquired, 0 if not. 

        """
        p = self.p
        if p['pslabel'] == 'ge3d_elliptical':
            bits = 32  # original encoding, see spinsights
            slice_shape = (int(p['nv']),int(p['nv2']))
            skiptensor = np.zeros(slice_shape)
            skipint = [int(x) for x in self.skipint]
            skipint_bin_vals = [str(np.binary_repr(d,bits)) for d in skipint]
            skipint_bin_vals = ''.join(skipint_bin_vals)
            skipint_bin_array = np.array([int(i) for i in skipint_bin_vals])
            skiptensor = np.reshape(skipint_bin_array, slice_shape,order='f')
            return skiptensor
            

        else:
            raise(Exception('only ge3d_elliptical implemented.'))

class SkipintGenerator():
    """Class to handle skipint parameter generation for compressed sensing

    Currently dynamic indexing is not supported. default kspace mask shape is:

                kspace_mask = np.ndarray([phase, read, slice, time])

    Methods:

        generate_kspace_mask(): 
            Makes k-space mask in the form of numpy array, containing 0s and 1s
        generate_skipint():
            Makes skipint parameter, which is an encoded (compressed) version
            of the mask
    """
    def __init__(self,\
                procpar=None,\
                shape=None,\
                seqdim=None,\
                reduction=4,\
                center=1/16,\
                distribution='gauss',\
                verbose=False):
        """Set up shape of mask

        Args:
            procpar (str): path to procpar file
            shape (tuple): shape of kspace, excluding receivers
            seqdim [2,3]: sequence type, 2dim or 3dim
            reduction (int) : subsampling ration
            center (float) : keep center ratio
            distribution (str) : gauss or random #TODO
            verbose (boolean)

        """
        if procpar != None:

            p = vj.io.ProcparReader(procpar).read()
            conf = vj.config

            if '2D' in p['apptype']:
                dim = 2
            elif '3D' in p['apptype']:
                dim = 3
            # TODO repetitions should handle diffusion and arrayed tr as well
            try:
                reps = int(p['images'])
            except:
                reps = 1
            try:
                echos = int(p['ne'])
            except:
                echos = 1

            shape = [1,1,1,1]
            shape[conf['ro_dim']-1] = int(p['np'])//2
            shape[conf['pe_dim']-1] = int(p['nv'])
            shape[conf['et_dim']-1] = echos * reps
            if dim ==2 :            
                shape[conf['slc_dim']-1] = int(p['ns'])
            elif dim == 3:
                shape[conf['pe2_dim']-1] = int(p['nv2'])

            self.shape = shape
            self.dim = dim
            self.reduction = reduction
            self.center = center
            self.conf = conf
            self.verbose = verbose

    def generate_kspace_mask(self):
        """Makes a kspace mask according to init params

        Returns:

            kspace_mask (np.ndarray) kspace mask of values 0 or 1
        """
        def _make_2d_mask(num):
            """make 2d mask"""
            mask2d = np.zeros((self.shape[0], self.shape[1]),dtype=int)
            weights = list(gaussian(self.shape[0], self.shape[0]/6))
            points = np.zeros(self.shape[0],dtype=int)
            center_ones = [int(points.shape[0]/2*(1-self.center)),\
                            int(points.shape[0]/2*(1+self.center))]
            center_ones_num = center_ones[1] - center_ones[0]
            points[center_ones[0]:center_ones[1]] = 1
            num = num - center_ones_num
            if num < 0:
                raise(Exception('No remaining points to choose: too high\
                                 reduction, or too much points in center'))
        
            indices_available = [i for i, x in enumerate(points) if x == 0]
            weights_available = [weights[i] for i in indices_available]
            weights_available_normalized = [float(i)/sum(weights_available) \
                                            for i in weights_available]
            choices = np.random.choice(indices_available,\
                                        int(num),\
                                        p=weights_available_normalized,\
                                        replace=False)

            for index in choices:
                points[index] = 1

            for i in range(self.shape[1]):

                mask2d[:,i] = np.array(points)

            return mask2d

        def _make_3d_mask(num):

            def gaussian_2d(x, y, stddev_x, stddev_y):

                gx = gaussian(x, stddev_x)
                gy = gaussian(y, stddev_y)
                weights = np.outer(gx,gy.T)
                return weights

            def make_circle_mask(x, y, rx, ry, cx=None, cy=None):

                center_ones = np.zeros([x,y])
                grid = np.ogrid[-x/2 : x/2, -y/2 : y/2]
                mask = grid[0]**2 + grid[1]**2 <= rx*ry
                center_ones[mask] = 1
                return center_ones

            mask3d = np.zeros(self.shape,dtype=int)[...,0]
            points_2d = np.zeros([self.shape[0], self.shape[1]],dtype=int)
            weights_2d = gaussian_2d(self.shape[0], self.shape[1],\
                            self.shape[0]/6, self.shape[1]/6)
            radius_x = self.shape[0]*self.center
            radius_y = self.shape[1]*self.center
            center_ones = make_circle_mask(self.shape[0],
                                            self.shape[1],
                                            radius_x,
                                            radius_y)
            # reshaping weights, points into 1d array
            weights_1d = np.reshape(weights_2d, weights_2d.size)
            points_1d = np.reshape(center_ones, points_2d.size)
            num = num - np.count_nonzero(center_ones == 1)
            if num < 0:
                raise(Exception('No remaining points to choose: too high\
                                 reduction, or too much points in center'))
        
            indices_available = [i for i, x in enumerate(points_1d) if x == 0]
            weights_available = [weights_1d[i] for i in indices_available]
            weights_available_normalized = [float(i)/sum(weights_available) \
                                            for i in weights_available]
            choices = np.random.choice(indices_available,\
                                        int(num),\
                                        p=weights_available_normalized,\
                                        replace=False)

            # what is this?!
            for index in choices:
                points_1d[index] = 1

            points_2d = np.reshape(points_1d, points_2d.shape)

            for i in range(self.shape[2]):

                mask3d[:,:,i] = np.array(points_2d)

            return mask3d

        # --------------------- generate k mask MAIN ------------------------

        kspace_mask = np.zeros(self.shape,dtype=int)
        
        if self.dim == 2:

            num = np.floor(self.shape[0]/self.reduction)

            for t in range(self.shape[self.conf['et_dim']-1]):

                for slc in range(self.shape[self.conf['slc_dim']-1]):

                    kspace_mask[:,:,slc,t] = _make_2d_mask(num)

            if self.verbose:
                print('Mask ready, shape: {}'.format(self.kspace_mask.shape))

        elif self.dim ==3:

            num = np.floor(self.shape[self.conf['pe_dim']-1]*\
                            self.shape[self.conf['pe2_dim']-1]/\
                            self.reduction)

            for t in range(self.shape[self.conf['et_dim']-1]):

                kspace_mask[:,:,:,t] = _make_3d_mask(num)
            #TODO currenty this is a debugging hack. better do it properly 
            kspace_mask = np.swapaxes(kspace_mask,1,2)
        else:
            raise(Exception('wrong dimension maybe'))

        return kspace_mask
    
    def generate_skipint(self,kmask):
        """Makes skipint parameter
        
        Args:
            kmask (np.ndarray) mask, generated by generate_kspace_mask()
        Returns:
            skipint
        """
        pass

