import copy
import vnmrjpy as vj
import numpy as np
import matplotlib.pyplot as plt

class Lmafit():
    """Low-rank matrix fitting algorithm

    Fills missing matrix elements by low rank approximation

    ref.: paper
    """
    def __init__(self,init_data,\
                known_data='NOT GIVEN',\
                tol=None,\
                k=None,\
                rank_strategy=None,\
                verbose=False,\
                realtimeplot=False):
        """Initialize Lmafit, get defaults from vnmrjpy config

        Args:
            init_data (np.ndarray) : matrix to complete. unkown elements can
                    be approximated beforehand
            known_data (np.ndarray) : matrix of the same shape as init_data 
                    with only the known elements, rest are zero
            tol (float) : tolerance of fitting
            rank_strategy (str) : increase or decrease rank

            verbose
            realtimeplot
        """
        conf = vj.config
        if tol == None:
            tol = conf['lmafit_tol']
        if rank_strategy == None:
            rank_strategy = conf['lmafit_rank_strategy']
        if k == None:
            k = conf['lmafit_start_rank']
        if (type(known_data) == str and known_data == 'NOT GIVEN'):
            if init_data[init_data == 0].size < init_data.size / 10:
                raise(Exception('Known data not given'))
            known_data = copy.deepcopy(init_data)

        self.verbose = verbose
        (m,n) = init_data.shape

        datamask = copy.deepcopy(known_data)
        datamask[datamask != 0+0*1j] = 1
        datanrm = np.linalg.norm(init_data,'fro')
        # init
        #Z = np.matrix(init_data)
        Z = init_data
        #X = np.matrix(np.zeros((m,k),dtype='complex64'))
        #TODO check removed matrix ok
        X = np.zeros((m,k),dtype='complex64')
        Y = np.eye(k,n,dtype='complex64')
        Res = np.multiply(init_data,datamask) - known_data
        res = datanrm
        reschg_tol = 0.5*tol
        # parameters for alf
        alf = 0
        increment = 0.5
        #rank estimation parameters
        itr_rank = 0
        minitr_reduce_rank = 5
        maxitr_reduce_rank = 50
        tau_limit = 10
        rank_incr = 3
        rank_max = 50

        self.realtimeplot = realtimeplot
        if realtimeplot == True:
            self.rtplot = vj.util.RealTimeImshow(np.absolute(init_data))


        self.initpars = (init_data,known_data,\
                        m,n,k,tol,rank_strategy,datanrm,\
                        Z,X,Y,Res,res,reschg_tol,alf,increment,itr_rank,\
                        minitr_reduce_rank,maxitr_reduce_rank,tau_limit,\
                        datamask,rank_incr,rank_max)

    def solve(self,max_iter=100):
        """Main iteration

        Completed matrix: Z = X*Y

        Returns :

            X (np.matrix)
            Y (np.matrix) 
            out_list [] : some list of helper outputs
        """
        def rank_check(R,reschg,tol):
            
            # diag = np.diag(R)
            # d_hat = [diag[i]/diag[i+1] for i in range(len(diag)-1)]
            # tau = (len(diag)-1)*max(d_hat)/(sum(d_hat)-max(d_hat))

            if reschg < 10*tol:
                ind_string = 'increase'
            else:
                ind_string = 'stay'
            return ind_string

        def increase_rank(X,Y,Z,rank_incr,rank_max):
            
            k = X.shape[1]
            k_new = min(k+rank_incr,rank_max)

            m = X.shape[0]
            n = Y.shape[1]
            #X_new = np.matrix(np.zeros((m,k_new),dtype='complex64'))
            #Y_new = np.matrix(np.eye(k_new,n,dtype='complex64'))            
            X_new = np.zeros((m,k_new),dtype='complex64')
            Y_new = np.eye(k_new,n,dtype='complex64')          
            X_new[:,:k] = X
            Y_new[:k,:] = Y
            Z_new = X.dot(Y)
            return X_new, Y_new, Z_new

        # -------------------INIT------------------------

        (data,known_data,m,n,k,tol,rank_strategy,datanrm,\
        Z,X,Y,Res,res,reschg_tol,alf,increment,itr_rank,\
        minitr_reduce_rank,maxitr_reduce_rank,tau_limit,\
                    datamask, rank_incr,rank_max) = self.initpars

        # --------------MAIN ITERATION--------------------
        objv = np.zeros(max_iter)
        RR = np.ones(max_iter)

        for iter_ in range(max_iter):
            itr_rank += 1

            X0 = copy.deepcopy(X)
            Y0 = copy.deepcopy(Y)
            Res0 = copy.deepcopy(Res)
            res0 = copy.deepcopy(res)
            Z0 = copy.deepcopy(Z)
            X = Z.dot(Y.conj().T)
            X, R = np.linalg.qr(X)
            Y = X.conj().T.dot(Z)
            Z = X.dot(Y)

            Res = np.multiply(known_data-Z,datamask)
            res = np.linalg.norm(Res,'fro')
            relres = res / datanrm
            ratio = res / res0
            reschg = np.abs(1-ratio)
            RR[iter_] = ratio
            # adjust alf
            if self.verbose == True:

                print('ratio : {}; rank : {}; reschg : {}, alf : {}'\
                        .format(ratio,X.shape[1],reschg, alf))

            if ratio >= 1.0:
                increment = np.max([0.1*alf,0.1*increment])
                X = copy.deepcopy(X0)
                Y = copy.deepcopy(Y0)
                Res = copy.deepcopy(Res0)
                res = copy.deepcopy(res0)
                relres = res / datanrm
                alf = 0
                Z = copy.deepcopy(Z0)
            elif ratio > 0.7:
                increment = max(increment,0.25*alf)
                alf = alf + increment 
            objv[iter_] = relres
            # check stopping
            if ((reschg < reschg_tol) and ((itr_rank > minitr_reduce_rank) \
                                    or (relres < tol))):
                if self.verbose == True:
                    print('Stopping crit achieved')
                break

            # rank adjustment
            rankadjust = rank_check(R,reschg,tol)
            if rankadjust == 'increase':
                X,Y,Z = increase_rank(X,Y,Z,rank_incr,rank_max)

            Zknown = known_data + alf*Res
            Z = Z - np.multiply(Z,datamask) + Zknown

            if self.realtimeplot == True:
                self.rtplot.update_data(np.absolute(Z))

        obj = objv[:iter_]

        return X, Y, [obj, RR, iter_, relres, reschg] 
