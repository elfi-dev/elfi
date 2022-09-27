"""Calculate the gaussian copula density used in the semiBsl method."""

import logging
import math

import numpy as np
from scipy.stats import norm

from elfi.methods.bsl.cov_warton import corr_warton

logger = logging.getLogger(__name__)


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


def gaussian_copula_density(rho_hat, u, whitening=None, eta_cov=None):
    """Log gaussian copula density for summary statistic likelihood.

    Parameters
    ----------
    rho_hat : np.array,
        The estimated correlation matrix for the simulated summaries
    u : np.array
        The CDF of the observed summaries for the KDE using simulated summaries
    whitening : np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller batch_size).
    eta_cov : np.array of shape (m x m) - m = num of summary statistics
        The sample covariance for the simulated etas used in wsemiBsl
    Returns
    -------
    logpdf of gaussian copula

    """
    eta = norm.ppf(u)  # inverse normal CDF -> eta ~ N(0,1)
    if whitening is not None:  # logic for wsemiBsl
        eta = np.matmul(whitening, eta)

        rho_hat_sigma = np.matmul(np.matmul(whitening, eta_cov),
                                  np.transpose(whitening))
        rho_hat_sigma_diag = np.diag(np.sqrt(np.diag(rho_hat_sigma)))
        rho_hat = np.matmul(rho_hat_sigma_diag,
                            np.matmul(rho_hat, rho_hat_sigma_diag))

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
        mat = np.linalg.inv(rho)
    except np.linalg.LinAlgError:
        logger.warning('Unable to invert rho, the estimated correlation matrix'
                       'for the simulated summaries.')
        return -math.inf

    # this is the same as np.dot(np.dot(eta, np.subtract(mat, np.eye(dim))), eta) but compatible
    # with whitened eta and mat
    mat_res = np.dot(np.dot(np.transpose(eta), mat), eta) - np.sum(eta**2)
    res = -0.5 * (logdet + mat_res)
    return res
