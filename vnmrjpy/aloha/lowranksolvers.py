import copy
import numpy as np
import vnmrjpy as vj
import sys
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
"""
Collection of solvers for low rank matrix completion
"""
def svt(A,known_data=None,tau=None, delta=None, epsilon=1e-4,max_iter=100,\
        realtimeplot=False):
    """Low rank matrix completion algorihm with singular value thresholding

    Uses singular value decomposition then throws away the small singular
    values each iteration. Defaults are used as in the reference paper

    Ref.: Candes paper

    Args:
        A (np.ndarray) -- 2 dimensional numpy array to be completed
        known_data (np.ndarray)-- known elements of A in the same shape,
                                    unknown are 0
        tau (float)
        delta (float)
        epsilon (float)
        max_iter (int) -- maximum number of iterations
        realtimeplot (boolean) -- convenience option to plot the result each 
                                iteration
    Return:
        A_filled (np.ndarray) -- completed 2dim numpy array
    """
    Y = np.zeros_like(A)
    mask = np.zeros_like(A)
    mask[known_data != 0] = 1

    if tau == None:
        tau = 5*np.sum(A.shape)/2
    if delta == None:
        delta = 1.2*np.prod(A.shape)/np.sum(mask)
    if realtimeplot == True:
        rtplot = vj.util.RealTimeImshow(np.abs(A))
    
    # start iteration
   
    for _ in range(max_iter): 
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        S = np.maximum(S-tau, 0)
        X = np.linalg.multi_dot([U, np.diag(S), V])
        Y = Y + delta*mask*(A-X)
    
        rel_recon_error = np.linalg.norm(mask*(X-A)) / \
                            np.linalg.norm(mask*A)

        if _ % 100 == 0:
            sys.stdout.flush()
            print(rel_recon_error)
            pass
        if rel_recon_error < epsilon:
            break
        if realtimeplot == True:
            rtplot.update_data(np.absolute(X))

    return X

def lmafit():

    pass

def admm():
    
    pass

#Deprecate --------------------------------------------------------------------
class SingularValueThresholding():
    """
    Matrix completion by singular value soft thresholding
    """
    def __init__(self,A,tau=None, delta=None, epsilon=1e-4, max_iter=10,\
                realtimeplot=True):

        self.A = A
        self.Y = np.zeros_like(A)
        self.max_iter = max_iter
        self.epsilon = epsilon
        mask = copy.deepcopy(A)
        mask[mask != 0] = 1
        self.mask = mask 
        if tau == None:
            self.tau = 5*np.sum(A.shape)/2
        else:
            self.tau = tau
        if delta == None:
            self.delta = 1.2*np.prod(A.shape)/np.sum(self.mask)
        else:
            self.delta = delta

        if realtimeplot == True:
            self.rtplot = vj.util.RealTimeImshow(np.absolute(A))
            self.realtimeplot = realtimeplot

    def solve(self):
        """Main iteration, returns the completed matrix"""
        for _ in range(self.max_iter):

            U, S, V = np.linalg.svd(self.Y, full_matrices=False)
            S = np.maximum(S-self.tau, 0)
            X = np.linalg.multi_dot([U, np.diag(S), V])
            self.Y = self.Y + self.delta*self.mask*(self.A-X)
        
            rel_recon_error = np.linalg.norm(self.mask*(X-self.A)) / \
                                np.linalg.norm(self.mask*self.A)

            if _ % 100 == 0:
                sys.stdout.flush()
                print(rel_recon_error)
                pass
            if rel_recon_error < self.epsilon:
                break
            if self.realtimeplot == True:
                self.rtplot.update_data(np.absolute(X))
        return X
