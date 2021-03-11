import numpy as np
import math

def d_laplace(x, rate=1):
    n = len(x)
    return n * math.log(rate/2) - rate * np.sum(np.abs(x))

def slice_gamma_mean(self, ssx, loglik, gamma=None, tau=1, w=1, std=None,
                     maxiter=1000):
    def log_gamma_prior(x): # TODO: refactor?
        d_laplace(x, rate = 1 / tau)

    # if
    sample_mean = ssx.mean(0)
    # print('sample_mean', sample_mean)
    sample_cov = np.cov(np.transpose(ssx))
    std = np.std(ssx, axis=0)



    pass
