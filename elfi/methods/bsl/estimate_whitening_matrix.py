import numpy as np
from scipy import linalg
from scipy.stats import norm

def estimate_whitening_matrix(ssx, method=None):
    # if method == "semiBsl": # TODO: CHECK !!! Whiten on what for wsemiBsl
        # ssx = norm.cdf(ssx)

    mu = np.mean(ssx, axis=0) # TODO: Assumes ssx dims (handling in init)
    std = np.std(ssx, axis=0)
    ns, n = ssx.shape
    mu_mat = np.tile(np.array([mu]), (ns, 1))
    std_mat = np.tile(np.array([std]), (ns, 1))
    ssx_std = (ssx - mu_mat) / std_mat
    cov_mat = np.cov(np.transpose(ssx_std)) # TODO: Assumes ssx dims
    w, v = linalg.eig(cov_mat)
    diag_w = np.diag(np.power(w, -0.5)).real.round(8)
    w_pca = np.dot(diag_w, v.T)
    return w_pca