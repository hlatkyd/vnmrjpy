import copy
import numpy as np
import vnmrjpy as vj
import sys
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
"""
Collection of solvers for low rank matrix completion
"""

class SingularValueThresholding():
    """
    Matrix completion by singular value soft thresholding
    """
    def __init__(self,A,tau=None, delta=None, epsilon=1e-4, max_iter=1000,\
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
