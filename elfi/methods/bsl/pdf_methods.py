"""Implements different BSL methods that estimate the approximate posterior."""

import logging
import math
from functools import partial

import numpy as np
import scipy.stats as ss
from scipy.special import loggamma
from sklearn.covariance import graphical_lasso

from elfi.methods.bsl.cov_warton import corr_warton, cov_warton
from elfi.methods.bsl.gaussian_copula_density import gaussian_copula_density
from elfi.methods.bsl.gaussian_rank_corr import gaussian_rank_corr as grc

logger = logging.getLogger(__name__)


def standard_likelihood(shrinkage=None, penalty=None, whitening=None, standardise=False):
    """Return gaussian_syn_likelihood with selected setup.

    Parameters
    ----------
    see gaussian_syn_likelihood

    Returns
    -------
    callable

    """
    return partial(gaussian_syn_likelihood, shrinkage=shrinkage, penalty=penalty,
                   whitening=whitening, standardise=standardise)


def unbiased_likelihood():
    """Return gaussian_syn_likelihood_ghurye_olkin.

    Returns
    -------
    callable

    """
    return gaussian_syn_likelihood_ghurye_olkin


def semiparametric_likelihood(shrinkage=None, penalty=None, whitening=None):
    """Return semi_param_kernel_estimate with selected setup.

    Parameters
    ----------
    see semi_param_kernel_estimate

    Returns
    -------
    callable

    """
    return partial(semi_param_kernel_estimate, shrinkage=shrinkage, penalty=penalty,
                   whitening=whitening)


def robust_likelihood(adjustment):
    """Return syn_likelihood_misspec with selected setup.

    Parameters
    ----------
    see syn_likelihood_misspec

    Returns
    -------
    callable

    """
    return partial(syn_likelihood_misspec, adjustment=adjustment)


def gaussian_syn_likelihood(ssx, ssy, shrinkage=None, penalty=None, whitening=None,
                            standardise=False):
    """Calculate the posterior logpdf using the standard synthetic likelihood.

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x.
    ssy : np.array
        Observed summaries.
    shrinkage : str, optional
        The shrinkage method to be used with the penalty parameter.
    penalty : float, optional
        The penalty value to used for the specified shrinkage method.
        Must be between zero and one when using shrinkage method "Warton".
    whitening :  np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller simulation count).
    standardise : bool, optional
        Used with shrinkage method "glasso".

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssy = np.squeeze(ssy)
    if whitening is not None:
        ssy = np.matmul(whitening, ssy)
        ssx = np.matmul(ssx, np.transpose(whitening))  # decorrelated sim sums

    sample_mean = ssx.mean(0)
    sample_cov = np.atleast_2d(np.cov(ssx, rowvar=False))

    if shrinkage == 'glasso':
        if standardise:
            std = np.sqrt(np.diag(sample_cov))
            ssx = (ssx - sample_mean) / std
            sample_cov = np.atleast_2d(np.cov(ssx, rowvar=False))
        gl = graphical_lasso(sample_cov, alpha=penalty, max_iter=200)
        # NOTE: able to get precision matrix here as well
        sample_cov = gl[0]

    if shrinkage == 'warton':
        sample_cov = cov_warton(sample_cov, 1-penalty)

    try:
        loglik = ss.multivariate_normal.logpdf(
            ssy,
            mean=sample_mean,
            cov=sample_cov,
            )
    except np.linalg.LinAlgError:
        logger.warning('Unable to compute logpdf due to poor sample cov.')
        loglik = -math.inf

    return np.array([loglik])


def gaussian_syn_likelihood_ghurye_olkin(ssx, ssy):
    """Calculate the unbiased posterior logpdf.

    Uses the unbiased estimator of the synthetic likelihood.

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x.
    ssy : np.array
        Observed summaries.

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    n, d = ssx.shape
    mu = np.mean(ssx, 0)
    Sigma = np.cov(np.transpose(ssx))
    ssy = ssy.reshape((-1, 1))
    mu = mu.reshape((-1, 1))

    psi = np.subtract((n - 1) * Sigma,
                      (np.matmul(ssy - mu, np.transpose(ssy - mu))
                      / (1 - 1/n)))

    try:
        _, logdet_sigma = np.linalg.slogdet(Sigma)
        _, logdet_psi = np.linalg.slogdet(psi)
        A = wcon(d, n-2) - wcon(d, n-1) - 0.5*d*math.log(1 - 1/n)
        B = -0.5 * (n-d-2) * (math.log(n-1) + logdet_sigma)
        C = 0.5 * (n-d-3) * logdet_psi
        loglik = -0.5*d*math.log(2*math.pi) + A + B + C
    except np.linalg.LinAlgError:
        logger.warning('Unable to compute logpdf due to poor sample cov.')
        loglik = -math.inf

    return np.array([loglik])


def semi_param_kernel_estimate(ssx, ssy, shrinkage=None, penalty=None, whitening=None):
    """Calculate the semiparametric posterior logpdf.

    Uses the semi-parametric log likelihood
    of An, Z. (2020).

    References
    ----------
    An, Z., Nott, D. J., and Drovandi, C. C. (2020).
    Robust Bayesian Synthetic Likelihood via a semi-parametric approach.
    Statistics and Computing, 30:543557.

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x.
    ssy : np.array
        Observed summaries.
    shrinkage : str, optional
        The shrinkage method to be used with the penalty parameter.
    penalty : float, optional
        The penalty value to used for the specified shrinkage method.
        Must be between zero and one when using shrinkage method "Warton".
    whitening :  np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller simulation count).

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssy = np.squeeze(ssy)
    n, ns = ssx.shape

    logpdf_y = np.zeros(ns)
    y_u = np.zeros(ns)
    sim_eta = np.zeros((n, ns))  # only used for wsemibsl
    eta_cov = None
    for j in range(ns):
        ssx_j = ssx[:, j].flatten()
        y = ssy[j]

        # NOTE: bw_method - "silverman" is being used here is slightly
        #       different than "nrd0" - silverman's rule of thumb in R.
        kernel = ss.gaussian_kde(ssx_j, bw_method="silverman")
        logpdf_y[j] = kernel.logpdf(y)

        y_u[j] = kernel.integrate_box_1d(np.NINF, y)
        y_u[j] = min(1, y_u[j])  # fix numerical errors, CDF values cannot exceed 1

        if whitening is not None:
            # TODO? Commented out very inefficient for large simulation count
            # sim_eta[:, j] = [ss.norm.ppf(kernel.integrate_box_1d(np.NINF,
            #                                                      ssx_i))
            #                  for ssx_i in ssx_j]
            sim_eta[:, j] = ss.norm.ppf(ss.rankdata(ssx_j)/(n+1))

    rho_hat = grc(ssx)

    if whitening is not None:
        sim_eta_trans = np.matmul(sim_eta, np.transpose(whitening))
        eta_cov = np.cov(np.transpose(sim_eta))
        rho_hat = grc(sim_eta_trans)

    if shrinkage == "glasso":
        # convert from correlation matrix -> covariance
        sample_cov = np.cov(ssx, rowvar=False)
        std = np.sqrt(np.diag(sample_cov))
        sample_cov = np.outer(std, std) * rho_hat
        # graphical lasso
        gl = graphical_lasso(sample_cov, alpha=penalty)
        sample_cov = gl[0]
        # convert from covariance -> correlation matrix
        std = np.sqrt(np.diag(sample_cov))
        rho_hat = np.outer(1/std, 1/std) * sample_cov

    if shrinkage == "warton":
        rho_hat = corr_warton(rho_hat, 1-penalty)

    gaussian_logpdf = gaussian_copula_density(rho_hat, y_u, whitening, eta_cov)
    pdf = gaussian_logpdf + np.sum(logpdf_y)

    return np.array([pdf])


def syn_likelihood_misspec(ssx, ssy, gamma, adjustment):
    """Calculate the posterior logpdf using the robust synthetic likelihood.

    Uses mean or variance adjustment to compensate for model misspecification.

    References
    ----------
    D. T. Frazier and C. Drovandi (2019).
    Robust Approximate Bayesian Inference with Synthetic Likelihood,
    J. Computational and Graphical Statistics, 30(4), 958-976.
    https://doi.org/10.1080/10618600.2021.1875839

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x
    ssy : np.array
        Observed summaries.
    gamma : np.array
        Adjustment parameter.
    adjustment : str
        String name of type of robust BSL. Can be either "mean" or "variance".

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssy = np.squeeze(ssy)
    sample_mean = ssx.mean(0)
    sample_cov = np.cov(ssx, rowvar=False)
    std = np.sqrt(np.diag(sample_cov))

    if adjustment == "mean":
        sample_mean = sample_mean + std * gamma

    if adjustment == "variance":
        sample_cov = sample_cov + np.diag((std * gamma) ** 2)

    try:
        loglik = ss.multivariate_normal.logpdf(
            ssy,
            mean=sample_mean,
            cov=sample_cov,
            )
    except np.linalg.LinAlgError:
        logger.warning('Unable to compute logpdf due to poor sample cov.')
        loglik = -math.inf

    return loglik


def wcon(k, nu):
    """Log of c(k, nu) from Ghurye & Olkin (1969).

    Args:
    k : int
    nu : int
    Returns:
    cc: float

    References
    ----------
    S. G. Ghurye, Ingram Olki.
    "Unbiased Estimation of Some Multivariate Probability Densities and
    Related Functions,".
    The Annals of Mathematical Statistics,
    Ann. Math. Statist. 40(4), 1261-1271, (August, 1969)

    """
    loggamma_input = [0.5*(nu - x) for x in range(k)]

    cc = -k * nu / 2 * math.log(2) - k*(k-1)/4*math.log(math.pi) - \
        np.sum(loggamma(loggamma_input))
    return cc
