from scipy import stats as sc
import numpy as np


def gaussian_rank_corr(x):
    """
    docstring
    """
    n, p = x.shape

    r = sc.rankdata(x, axis=0)
    rqnorm = sc.norm.ppf(r / (n+1))
    density = sum(sc.norm.ppf(np.divide(list(range(1, n+1)), (n+1))) ** 2)

    res = [ np.matmul(rqnorm[:, i], rqnorm[:, (i+1):(p+1)]) for i in range(p - 1)]
    res = np.concatenate(res).ravel()
    res = res / density

    return res
