import numpy as np
import vnmrjpy as vj
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
        # if no echo tie is specified get from procpar
        def _check_echo_times(tdim, echo_list):
            if len(echo_list) == tdim and tdim != 0:
                return True
            else:
                return False

        def _get_noise_mean(img_data):
            """Find mean noise by making a histogram"""

            # def constants
            bins = 100

            imgmax = np.max(img_data)
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

        # sanity check for fitfunc

        if fitfunc in ['3param_exp']:
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
        self.data = self.mask * self.data
        self.noise_mean = _get_noise_mean(img_data)

    def fit(self): 
        """Actual fitting method

        Return:
            (*fitted parameters)        
        """
        def fit_3param_exp(data,echo_times, mask):

            def _3param_exp(te,A,B,T2):
                return  A*np.exp(-te/T2) + B

            t2map = np.zeros_like(data)
            m0map = np.zeros_like(data)
            const = np.zeros_like(data)
            
            nm = self.noise_mean
            for x in range(data.shape[0]):
                for y in range(data.shape[1]):
                    for z in range(data.shape[2]):
                        print(mask[x,y,z,0])
                        if mask[x,y,z,0] == 0:
                            continue
                        else:
                            data_1d = data[x,y,z,:]
                            (m0,c,t2), cov  = curve_fit(_3param_exp,\
                                                        data_1d,\
                                                        self.echo_times,\
                                                        p0=(0.03,1000,nm),\
                                                        sigma=nm*2)
                            m0map[x,y,z] = m0
                            t2map[x,y,z] = t2
                            print(t2*1000)
                            const[x,y,z] = c
            return (t2map, m0map, const)

        if self.fitfunc == '3param_exp':        
            
            (t2map, m0map, const) = fit_3param_exp(\
                                    self.data,self.echo_times,self.mask)

            return t2map














