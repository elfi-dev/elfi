""" """

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import mahalanobis
import math
from elfi.methods.bsl.cov_warton import cov_warton, corr_warton


#TODO: Where to place ?
def p2P(param, n_rows):
    """Construct a summetric matrix 1s on the diagonal from the given
    parameter vector

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
    return P


def P2p(P, dim):
    return list(P[np.triu_indices(dim, 1)])


def gaussian_copula_density(rho_hat, u, penalty=None, whitening=None, whitening_eta=None, eta_cov=None):
    """[summary]

    Args:
        rho_hat ([type]): [description]
        u ([type]): [description]
        whitening ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    eta = norm.ppf(u)
    if whitening is not None:
        eta = np.matmul(whitening, eta)  # TODO: In process checking trans, order, etc

        # rho_hat = p2P(rho_hat, len(eta))
        # rho_hat = np.abs(rho_hat)
        r_warton = corr_warton(rho_hat, penalty)

        # TODO: rho_hat or eta_cov?
        rho_hat_sigma = np.matmul(np.matmul(whitening, eta_cov), np.transpose(whitening))
        rho_hat_sigma_diag = np.diag(np.sqrt(np.diag(rho_hat_sigma)))
        rho_hat =  np.matmul(rho_hat_sigma_diag,
                          np.matmul(r_warton, rho_hat_sigma_diag))
        # rho_hat_warton = corr_warton(rho_hat, penalty)
        # rho_hat_shrinkage_estimator = np.matmul(rho_hat_sigma_diag,
        #                  np.matmul(rho_hat_warton, rho_hat_sigma_diag))
        #!: TESTING

    dim = len(u)
    eta = np.array(eta).reshape(dim, 1)
    if any(np.isinf(eta)):
        return -math.inf

    if rho_hat.ndim == 1:
        rho = p2P(rho_hat, dim)
    else:
        rho = rho_hat
    
    _, logdet = np.linalg.slogdet(rho)  # don't need sign, only logdet
    if whitening is not None:
        _, logdet = np.linalg.slogdet(rho)

    try:
        mat = np.subtract(np.linalg.inv(rho), np.eye(dim))
        if whitening is not None:
            mat = np.linalg.inv(rho)
    except np.linalg.LinAlgError:
        return -math.inf

    mat_res = np.dot(np.dot(np.transpose(eta), mat), eta)
    mat_res = float(mat_res)
    res = -0.5 * (logdet + mat_res)
    return res