"""Calculate the gaussian copula density used in the semiBsl method."""

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


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

    eta = np.array(eta).reshape(-1, 1)
    if any(np.isinf(eta)):
        return -math.inf

    _, logdet = np.linalg.slogdet(rho_hat)  # don't need sign, only logdet

    try:
        mat = np.linalg.inv(rho_hat)
    except np.linalg.LinAlgError:
        logger.warning('Unable to invert rho, the estimated correlation matrix'
                       'for the simulated summaries.')
        return -math.inf

    # this is the same as np.dot(np.dot(eta, np.subtract(mat, np.eye(dim))), eta) but compatible
    # with whitened eta and mat
    mat_res = np.dot(np.dot(np.transpose(eta), mat), eta) - np.sum(eta**2)
    res = -0.5 * (logdet + mat_res)
    return res
