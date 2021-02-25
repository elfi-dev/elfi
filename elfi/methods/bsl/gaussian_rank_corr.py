from scipy import stats as sc
import numpy as np


def gaussian_rank_corr(x):
    """
    docstring
    """
    n, p = x.shape

    # print('n', n, 'p', p)

    r = sc.rankdata(x, axis=0)
    # print('rrrr', r)
    # print('r / (n+1)', np.mean(r / (n+1)))
    rqnorm = sc.norm.ppf(r / (n+1))
    # print('rqnorm', rqnorm)
    density = sum(sc.norm.ppf(np.divide(list(range(1, n+1)), (n+1))) ** 2)
    # print('density', density)
    # def f(x):
        # return np.matmul(rqnorm[:, x], rqnorm[:, list(range(x, p))]) / density

  
    res = [ np.matmul(rqnorm[:, i], rqnorm[:, (i+1):(p+1)]) for i in range(p - 1)]
    # print('initres', res)
    res = np.concatenate(res).ravel()
    # print('res', len(res))
    # print('res0', res)
    res = res / density
    # print('resfinal', res[0:100])
    # print('ress', len(res))
    # res_mat = np.zeros((p, p))
    # for i in range(p):
    #     # print('i', list(range(i, 4)))
    #     # print(res[i])
    #     res_mat[np.arange(i, p), i] = res[i]
    #     res_mat[i, np.arange(i, p)] = res[i]
    # print('res', res_mat)
    # print('p', p)
    # print('flatres', flatten(res))
    # print(1/0)
    # res = np.array(res).reshape()
    # print('res1', res[0:20])
    return res
