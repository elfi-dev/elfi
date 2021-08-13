from scipy import stats as ss
from .gaussian_copula_density import p2P
import numpy as np


def gaussian_rank_corr(x, vec=False):
    """
    docstring
    """
    n, p = x.shape[0:2]
    r = ss.rankdata(x, axis=0)
    rqnorm = ss.norm.ppf(r / (n+1))
    density = sum(ss.norm.ppf(np.divide(list(range(1, n+1)), (n+1))) ** 2)
    res = [np.matmul(rqnorm[:, i], rqnorm[:, (i+1):(p+1)]) for i in range(p - 1)]
    res = np.concatenate(res).ravel()
    res = res / density
    if not vec:
        res_mat = p2P(res, n_rows=p)
    return res_mat
