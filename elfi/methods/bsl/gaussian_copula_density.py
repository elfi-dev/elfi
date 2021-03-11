import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import mahalanobis
import math

#TODO: Where to place ?
def p2P(param, n_rows):
    """Construct a summetric matrix 1s on the diagonal from the given
    parameter vector (from p2P in the R copula package)

    Args:
        param ([type]): [description]

        Returns:
            [type]: [description]
    """
#     num_rows, _ = param.shape
    P = np.diag(np.zeros(n_rows))
    P[np.triu_indices(n_rows, 1)] = param
    P = np.add(P, np.transpose(P))
    np.fill_diagonal(P, 1)
    print('PPPP', P)
    return P

def P2p(P, dim):
    return list(P[np.triu_indices(dim, 1)])

def gaussian_copula_density(rho_hat, u, sd, whitening=None):
    # print('u', u)
    eta = norm.ppf(u)
    if whitening is not None:
        eta = np.matmul(whitening, eta)  # TODO: In process checking trans, order, etc
    print('upost', u)
    eta = norm.ppf(u)
    dim = len(u)
    eta = np.array(eta).reshape(dim, 1)
    print('etaeta', eta)
    eta[np.isposinf(eta)] = 1e+10 # TODO: CHECK WHY THESE CHECKS NEEDED
    eta[np.isneginf(eta)] = -1e+10 #TODO: Changed to +30...
    print('rho_hat.ndim', rho_hat.ndim)
    if rho_hat.ndim == 1:
        rho = p2P(rho_hat, dim)
    else:
        rho = rho_hat
    try:
        test = np.linalg.cholesky(rho)
    except Exception:
        raise("rho not SPD")
    det = np.linalg.det(rho)
    sign, logdet = np.linalg.slogdet(rho)
    mat = np.subtract(np.linalg.inv(rho), np.eye(dim))
    mat_res = np.dot(np.dot(np.transpose(eta), mat), eta)
    res = - (( logdet + mat_res) / 2)

    return res