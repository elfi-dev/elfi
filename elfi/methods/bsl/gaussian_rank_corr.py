"""Compute the gaussian rank correlation estimator."""

import numpy as np
import scipy.stats as ss


def p2P(param, n_rows):
    """Convert vector to symmetric matrix.

    Construct a symmetric matrix with 1s on the diagonal from the given
    parameter vector

    Parameters
    ----------
    param : np.array
    n_rows : int

    Returns
    -------
    P : np.array

    """
    P = np.diag(np.zeros(n_rows))
    P[np.triu_indices(n_rows, 1)] = param
    P = np.add(P, np.transpose(P))
    np.fill_diagonal(P, 1)
    return P


def gaussian_rank_corr(x):
    """Calculate the gaussian rank correlation matrix.

    Parameters
    ----------
    x : np.array
        Simulated summaries matrix

    Returns
    -------
    res_mat : np.array
        Gaussian rank correlation matrix

    """
    n, p = x.shape[0:2]
    r = ss.rankdata(x, axis=0)
    rqnorm = ss.norm.ppf(r / (n+1))
    density = np.sum(ss.norm.ppf(np.divide(list(range(1, n+1)), (n+1))) ** 2)
    res = [np.matmul(rqnorm[:, i], rqnorm[:, (i+1):(p+1)]) for i in range(p - 1)]
    res = np.concatenate(res).ravel()
    res = res / density
    res_mat = p2P(res, n_rows=p)
    return res_mat
