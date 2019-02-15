import numpy as np
import vnmrjpy as vj
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import copy
import sys

#TODO getting R2 error

class T2Fitter():

    def __init__(self,img_data, echo_times=[],\
                                procpar=None,\
                                skip_first_echo=False,\
                                mask=None,\
                                automask=False,\
                                fitmethod='scipy',\
                                fitfunc='3param_exp'):

        """Fit exponential to multiecho data 

        Args:
            img_data (np.ndarray) -- 4D imput data in image space
            echo_times (tuple) -- echo times corresponding to dim4 in img_data
            skip_first_echo (boolean) -- skips first datapoint if needed
            mask (np.ndarray) -- binary mask where fitting should be done
            automask (boolean) -- tries to detect noise and make proper mask
            fitmethod (str) -- 
            fitfunc (str) -- 
        """
        # if no echo time is specified get from procpar
        def _check_echo_times(tdim, echo_list):
            if len(echo_list) == tdim and tdim != 0:
                return True
            else:
                return False

        def _get_noise_mean(img_data):
            """Find mean noise by making a histogram"""

            # def constants
            bins = 100

            imgmax = np.max(np.array(img_data,dtype=np.float64))
            hist, bin_edges = np.histogram(img_data, bins=bins, range=(0,imgmax))
            # probably most of image is noise
            noise_peak = np.amax(hist)
            noise_mean = np.where(hist==noise_peak)[0][0]*bins
            return noise_mean

        def _automask(img_data):
            """Make mask by thresholding for mean noise and median filtering"""

            filter_size = 7
            
            noise_thresh = _get_noise_mean(img_data)*3
            mask = np.ones_like(img_data[...,0])
            mask[img_data[...,0] < noise_thresh] = 0
            mask = median_filter(mask, size=filter_size)
            # tile mask to same size as img_data
            mask = np.repeat(mask[...,np.newaxis],img_data.shape[-1],axis=-1)
            return mask

        # check data dimensions (should be 4)
        if len(img_data.shape) != 4:
            raise(Exception('Input data should have 4 dimensions')) 
        # sanity check for fitfunc
        if fitfunc in ['3param_exp','2param_exp']:
            pass
        else:
            raise(Exception('Cannot recognize fitfunc'))

        if _check_echo_times(img_data.shape[-1],echo_times) == False \
                                                    and procpar != None:
            ppdict = vj.io.ProcparReader(procpar).read()
            ne = int(ppdict['ne'])
            te = float(ppdict['te'])
            echo_times = [te * i for i in range(1,ne+1)]
        else:
            raise(Exception('Please specify procpar or echo times'))

        if skip_first_echo:
            self.data = img_data[:,:,:,1:]  # time dim is 4
            self.echo_times = echo_times[1:]
        else:
            self.data = img_data
            self.echo_times = echo_times
       
        if mask == None and automask == True:
            mask = _automask(self.data)
        else:
            mask = np.ones_like(self.data)
 
        self.mask = mask
        self.fitmethod = fitmethod  # some string option for later
        self.fitfunc = fitfunc
        self.data = np.array(self.mask * self.data,dtype=np.float64)
        self.echo_times = np.array(self.echo_times,dtype=np.float64)
        self.noise_mean = _get_noise_mean(img_data)

    def fit(self): 
        """Actual fitting method

        Return:
            (*fitted parameters)        
        """
        def fit_2param_exp(data,echo_times, mask):

            def _2param_exp(x,T2,A):
                y = A*np.exp(-x*1000/(T2*1000))
                return y

            t2map = np.zeros_like(data[...,0])
            m0map = np.zeros_like(data[...,0])
            et = self.echo_times
            nm = self.noise_mean
            y_err = np.asarray([nm/5 for i in range(len(echo_times))])
            print('Fitting 2 parameter exponential')
            for z in range(data.shape[2]):
                for y in range(data.shape[1]):
                    for x in range(data.shape[0]):
                        if mask[x,y,z,0] == 0:
                            continue
                        else:
                            try:
                                data_1d = data[x,y,z,:]
                                y0 = data_1d[0]
                                (t2,m0), cov  = curve_fit(_2param_exp,\
                                                et,\
                                                data_1d,\
                                                p0=(0.03,y0*1.5),\
                                                check_finite=False,\
                                                bounds=([0,0],[0.2,y0*3]),\
                                                method='trf')
                                                #sigma=y_err,\
                                                #absolute_sigma=True)
                                m0map[x,y,z] = m0
                                t2map[x,y,z] = t2
                                x_fit = np.linspace(0,et[-1],100)
                            except Exception:
                                t2map[x,y,z] = guess[0]
                                m0map[x,y,z] = guess[1]
                            except KeyboardInterrupt:
                                sys.exit(0)
            return (t2map, m0map)

        def fit_3param_exp(data,echo_times, mask):
            """Standard 3 parameter wxponential fitting"""

            def _3param_exp(x,T2,A,B):
                y = A*np.exp(-x*1000/(T2*1000)) + B
                return y

            t2map = np.zeros_like(data[...,0])
            m0map = np.zeros_like(data[...,0])
            const = np.zeros_like(data[...,0])
            
            et = self.echo_times
            nm = self.noise_mean
            # init and bounds for estimated params B, T2,
            #(A depends on intensity)
            y_err = np.asarray([nm for i in range(len(echo_times))])
            print('Fitting 3 parameter exponential')
            for z in range(data.shape[2]):
                for y in range(data.shape[1]):
                    for x in range(data.shape[0]):
                        if mask[x,y,z,0] == 0:
                            continue
                        else:
                            data_1d = data[x,y,z,:]
                            y0 = data_1d[0]
                            guess = (0.03,y0*1.5,nm)
                            bounds = ([0,y0/2,nm/2],[0.5,y0*2,nm*3])
                            try:
                                (t2,m0,c), cov  = curve_fit(_3param_exp,\
                                            et,\
                                            data_1d,\
                                            p0=guess,\
                                            check_finite=False,\
                                            bounds=bounds,\
                                            method='trf',\
                                            sigma=y_err,\
                                            absolute_sigma=True)
                                t2map[x,y,z] = t2
                                m0map[x,y,z] = m0
                                const[x,y,z] = c
                            except Exception:
                                t2map[x,y,z] = guess[0]
                                m0map[x,y,z] = guess[1]
                                const[x,y,z] = guess[2]
                            except KeyboardInterrupt:
                                sys.exit(0)
            return (t2map, m0map, const)

        #---------------------------Switch cases ------------------------------

        if self.fitfunc == '3param_exp':        
            
            return  fit_3param_exp(self.data,self.echo_times,self.mask)

        elif self.fitfunc == '2param_exp':        
            
            return fit_2param_exp(self.data,self.echo_times,self.mask)














