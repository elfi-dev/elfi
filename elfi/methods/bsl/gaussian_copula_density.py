"""Calculate the gaussian copula density used in the semiBsl method."""

import numpy as np
from scipy.stats import norm
import math
from elfi.methods.bsl.cov_warton import corr_warton


def p2P(param, n_rows):
    """Construct a symmetric matrix with 1s on the diagonal from the given
    parameter vector

    Parameters
    ----------
    param : np.array
    n_rows : int

    Returns
    -------
    P : np.array
    """
#     num_rows, _ = param.shape
    P = np.diag(np.zeros(n_rows))
    P[np.triu_indices(n_rows, 1)] = param
    P = np.add(P, np.transpose(P))
    np.fill_diagonal(P, 1)
    return P


# def P2p(P, dim):
#     """ """
#     return list(P[np.triu_indices(dim, 1)])


def gaussian_copula_density(rho_hat, u, penalty=None, whitening=None,
                            eta_cov=None):
    """log gaussian copula density for summary statistic likelihood.

    Parameters
    ----------
    rho_hat : np.array,
        The estimated correlation matrix for the simulated summaries
    u : np.array
        The CDF of the observed summaries for the KDE using simulated summaries
    penalty : float
        The warton shrinkage penalty (used with whitening)
    whitening :  np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller batch_size).
    eta_cov :np.array of shape (m x m) - m = num of summary statistics
        The sample covariance for the simulated etas used in wsemiBsl
    Returns
    -------
    logpdf of gaussian copula
    """

    eta = norm.ppf(u)  # inverse normal CDF -> eta ~ N(0,1)
    if whitening is not None:  # logic for wsemiBsl
        # refer to... for details #TODO!
        eta = np.matmul(whitening, eta)

        r_warton = corr_warton(rho_hat, penalty)

        rho_hat_sigma = np.matmul(np.matmul(whitening, eta_cov),
                                  np.transpose(whitening))
        rho_hat_sigma_diag = np.diag(np.sqrt(np.diag(rho_hat_sigma)))
        rho_hat = np.matmul(rho_hat_sigma_diag,
                            np.matmul(r_warton, rho_hat_sigma_diag))

    dim = len(u)
    eta = np.array(eta).reshape(-1, 1)
    if any(np.isinf(eta)):
        return -math.inf

    if rho_hat.ndim == 1:
        rho = p2P(rho_hat, dim)
    else:
        rho = rho_hat

    _, logdet = np.linalg.slogdet(rho)  # don't need sign, only logdet

    try:
        mat = np.subtract(np.linalg.inv(rho), np.eye(dim))
        # mat = np.linalg.inv(rho)
    except np.linalg.LinAlgError:
        return -math.inf

    mat_res = np.dot(np.dot(np.transpose(eta), mat), eta)
    mat_res = float(mat_res)
    res = -0.5 * (logdet + mat_res)
    return res
