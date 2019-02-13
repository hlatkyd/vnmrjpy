import os
import glob
import nibabel as nib
import numpy as np

class Ideal():
    """Fat fraction calculation for Dixon method with iterative least squares fitting.

    Ref.: dixon paper, ideal paper  
    """
    def __init__(self,kspace_data_list, ro_shift_list, freq=(0,1400)):
        """Fat fraction calculation for Dixon method with iterative least squares fitting.

        Args:
            kspace_data_list (list of np.ndarra) -- k-spaces with different
                                                    readout shifts
            ro_shift_list (tuple) -- readout shifts in seconds
            freq (tuple) -- frequency offsets in Hz of the Dixon components at 9.4T
        """

        self.data_list = kspace_data_list
        self.roshift_lsit = ro_shift_list
        self.freq = freq

    def fit(self, iterations, mask=None, fieldcorr=None, selfmask=False):
        """IDEAL fitting method
        
        Args.:
            iterations
            mask (np.ndarray) -- binary mask where fitting is performed

        Return:
        """

        def apply_field_correction(data, field, roshift):

            for k, shift in enumerate(roshift):
                data[k,...] = data[k,...]*np.exp(-1j*2*np.pi*field*shift)            
            return data

        def recalc_ro_final(ro, A, field, S):
        
            fullA = np.repeat(A,S.size).reshape(A.shape + S.shape,order='c')
            ro =  np.linalg.multi_dot((np.linalg.inv(np.dot(A.T,A)),A.T,S))
            return ro

        def make_Acd(roshift, freq):
            A = np.zeros((2*len(roshift),2*len(freq)))
            c = np.asarray([math.cos(2*np.pi*i*j) for i in freq for j in roshift])
            d = np.asarray([math.sin(2*np.pi*i*j) for i in freq for j in roshift])
            c = np.reshape(c,(len(freq),len(roshift)),order='c')
            d = np.reshape(d,(len(freq),len(roshift)),order='c')
            
            A[:len(roshift),0::2] = c.T
            A[len(roshift):,0::2] = d.T
            A[:len(roshift),1::2] = -d.T
            A[len(roshift):,1::2] = c.T 

            return A, c, d

        def fit_voxelvise(data1D, roshift, A, c, d, F0=0, freq=self.freq):
            """1d function to be used for apply_along_axis
    
            F0 is initial field
            freq is the Dixon components frequency shift
            """
            def _Sh_from_f(data1D, field, freq, roshift):

                Sh = data1D*np.exp(-1j*2*np.pi*field*np.asarray(roshift))
                Sre = np.asarray([np.real(Sh[i]) for i in range(Sh.shape[0])])
                Sim = np.asarray([np.imag(Sh[i]) for i in range(Sh.shape[0])])

                return np.concatenate((Sre, Sim),axis=0).T # return col vector
            
            def _ro_from_Sh(S,A):

                return np.linalg.multi_dot((np.linalg.inv(np.dot(A.T,A)),A.T,S))

            def _B_from_ro(ro, A, c, d, roshift, freq):

                B = np.zeros((A.shape[0],A.shape[1]+1))
                B[:,1:] = A
                rore = ro[0::2]
                roim = ro[1::2]
                gre = [2*np.pi*roshift[i]*(-rore[j]*d[j,i]-roim[j]*c[j,i]) \
                         for i in range(len(roshift)) for j in range(len(freq))] # g1N_re
                gre = np.sum(np.reshape(gre,(c.shape),order='F'),axis=0)
                #gre = np.reshape(gre,(c.shape),order='F')
                gim = [2*np.pi*roshift[i]*(rore[j]*c[j,i]-roim[j]*d[j,i]) \
                         for i in range(len(roshift)) for j in range(len(freq))] # g1N_re
                gim = np.sum(np.reshape(gim,(c.shape),order='F'),axis=0)
                #gre = np.reshape(gre,(c.shape),order='F')
            
                B[:len(gre),0] = gre
                B[len(gim):,0] = gim

                return B

            def _y_from_Sh(S,B):# y contains the error terms delta_field, delta_ro     

                return np.linalg.multi_dot((np.linalg.inv(np.dot(B.T,B)),B.T,S))

            def _Sh_from_B_y(B,y):
    
                return np.dot(B,y)

            # actual iteration starts here

            f = 0 # init fieldmap=0
            for i in range(iterations):

                Sh = _Sh_from_f(data1D,f,freq,roshift)
                ro = _ro_from_Sh(Sh,A)
                B = _B_from_ro(ro,A,c,d,roshift,freq)
                y = _y_from_Sh(Sh,B)
                f = np.asarray(f + y[0]) # recalculate field
                # make iteration stop on voxel basis
                if abs(y[0]) < 1:
                    break

            return np.concatenate([ro,y,np.atleast_1d(f)])

        """-------------------------IDEAL_fit MAIN-------------------------------------"""

        if self.proc_check_ok:
            print('dixonProc.ideal_fit() : Preprocess check OK')
        else:
            print('dixonProc.ideal_fit() : Preprocess was not done. \
                    Use dixonPreproc().run() first')
            return
        # data.shape : [roshift, read, phase, slice, time=1]
        # field.shape : [read, phase, slice, 1]

        data, field, roshift, header, affine = read_proc_dir()

        if fieldcorr == True: 

            data = apply_field_correction(data, field, roshift)

        A,c,d = make_Acd(roshift, self.freq)

        out = np.apply_along_axis(fit_voxelvise, 0, data, roshift, A, c, d, self.freq)

        if self.saving:
            self.save_results_to_nifti(out, affine, header)
        print('Done!')

        return out # data is the masked data

if __name__ == '__main__':
    
    study_dir = '/home/david/dev/dixon/s_2018080901'

    dp = dixonProc(study_dir)
    #dp.least_squares_fit()
    dp.ideal_fit(100)
