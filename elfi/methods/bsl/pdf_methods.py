"""Implements different BSL methods that estimate the approximate posterior."""

import logging
import math
from functools import partial

import numpy as np
import scipy.stats as ss
from scipy.special import loggamma
from sklearn.covariance import graphical_lasso

from elfi.methods.bsl.cov_warton import cov_warton
from elfi.methods.bsl.gaussian_copula_density import gaussian_copula_density
from elfi.methods.bsl.gaussian_rank_corr import gaussian_rank_corr as grc
from elfi.methods.bsl.slice_gamma_mean import slice_gamma_mean
from elfi.methods.bsl.slice_gamma_variance import slice_gamma_variance

logger = logging.getLogger(__name__)


def bsl_likelihood(shrinkage=None, penalty=None, whitening=None, standardise=False):
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


def semibsl_likelihood(shrinkage=None, penalty=None, whitening=None):
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


def misspec_likelihood(adjustment, tau=0.5, w=1, max_iter=1000):
    """Return syn_likelihood_misspec with selected setup.

    Parameters
    ----------
    see syn_likelihood_misspec

    Returns
    -------
    callable

    """
    return partial(syn_likelihood_misspec, adjustment=adjustment, tau=tau, w=w, max_iter=max_iter)


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
        The shrinkage method to be used with the penalty param. With "glasso"
        this corresponds with BSLasso and with "warton" this corresponds
        with wBsl.
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
        sample_cov = cov_warton(sample_cov, penalty)

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
        The shrinkage method to be used with the penalty param. With "glasso"
        this corresponds with BSLasso and with "warton" this corresponds
        with wBsl.
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
        kernel = ss.kde.gaussian_kde(ssx_j, bw_method="silverman")
        logpdf_y[j] = kernel.logpdf(y)

        y_u[j] = kernel.integrate_box_1d(np.NINF, y)

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
        sample_cov = np.cov(ssx, rowvar=False)
        std = np.sqrt(np.diag(sample_cov))
        # convert from correlation matrix -> covariance
        sample_cov = np.outer(std, std) * rho_hat
        sample_cov = np.atleast_2d(sample_cov)
        gl = graphical_lasso(sample_cov, alpha=penalty)
        sample_cov = gl[0]
        rho_hat = np.corrcoef(sample_cov)

    gaussian_logpdf = gaussian_copula_density(rho_hat, y_u,
                                              penalty, whitening,
                                              eta_cov)

    pdf = gaussian_logpdf + np.sum(logpdf_y)

    if whitening is not None:
        Z_y = ss.norm.ppf(y_u)
        pdf -= np.sum(ss.norm.logpdf(Z_y, 0, 1))

    pdf = np.nan_to_num(pdf, nan=np.NINF)

    return np.array([pdf])


def syn_likelihood_misspec(ssx, ssy, prev_state, adjustment="variance", tau=0.5,
                           w=1, max_iter=1000, random_state=None):
    """Calculate the posterior logpdf using the standard synthetic likelihood.

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x
    ssy : np.array
        Observed summaries.
    adjustment : str
        String name of type of misspecified BSL. Can be either "mean" or
        "variance".
    tau : float, optional
        Scale (or inverse rate) parameter for the Laplace prior distribution of
        gamma. Defaults to 1.
    w : float, optional
        Step size used for stepping out procedure in slice sampler.
    max_iter : int, optional
        The maximum numer of iterations for the stepping out and shrinking
        procedures for the slice sampler algorithm.

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    s1, s2 = ssx.shape
    ssy = np.squeeze(ssy)
    dim_ss = len(ssy)

    prev_iter_loglik = prev_state['loglikelihood']
    prev_sample_mean = prev_state['sample_mean']
    prev_sample_cov = prev_state['sample_cov']
    # first iter -> does not use mean/var - adjustment
    gamma = None
    if prev_iter_loglik is not None:
        gamma = prev_state['gamma']
        if gamma is None:
            if adjustment == "mean":
                gamma = np.repeat(0., dim_ss)
            if adjustment == "variance":
                gamma = np.repeat(tau, dim_ss)
        if adjustment == "mean":
            gamma, prev_iter_loglik = slice_gamma_mean(ssy,
                                                       loglik=prev_iter_loglik,
                                                       gamma=gamma,
                                                       sample_mean=prev_sample_mean,
                                                       sample_cov=prev_sample_cov,
                                                       tau=tau,
                                                       w=w,
                                                       max_iter=max_iter,
                                                       random_state=random_state
                                                       )
        if adjustment == "variance":
            gamma, prev_iter_loglik = slice_gamma_variance(ssy,
                                                           loglik=prev_iter_loglik,
                                                           gamma=gamma,
                                                           sample_mean=prev_sample_mean,
                                                           sample_cov=prev_sample_cov,
                                                           tau=tau,
                                                           w=w,
                                                           max_iter=max_iter,
                                                           random_state=random_state
                                                           )

    if s1 == dim_ss:  # obs as columns
        ssx = np.transpose(ssx)

    sample_mean = ssx.mean(0)
    sample_cov = np.cov(ssx, rowvar=False)
    std = np.sqrt(np.diag(sample_cov))

    if gamma is None:
        if adjustment == "mean":
            gamma = np.repeat(0., dim_ss)
        if adjustment == "variance":
            gamma = np.repeat(tau, dim_ss)

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

    return loglik, gamma, prev_iter_loglik


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
