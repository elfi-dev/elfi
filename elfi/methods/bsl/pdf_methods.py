
"""Implements different BSL methods that estimate the approximate posterior."""
# import scipy.optimize
import math

import numpy as np
import scipy.stats as ss
from scipy.special import loggamma
from sklearn.covariance import graphical_lasso

from elfi.methods.bsl.cov_warton import cov_warton
from elfi.methods.bsl.gaussian_copula_density import gaussian_copula_density
from elfi.methods.bsl.gaussian_rank_corr import gaussian_rank_corr as grc
from elfi.methods.bsl.slice_gamma_mean import slice_gamma_mean
from elfi.methods.bsl.slice_gamma_variance import slice_gamma_variance


def gaussian_syn_likelihood(*ssx, shrinkage=None, penalty=None,
                            whitening=None, standardise=False, observed=None,
                            **kwargs):
    """Calculate the posterior logpdf using the standard synthetic likelihood.

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x
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
        for heaving shrinkage to be applied (hence smaller batch_size).
    standardise : bool, optional
        Used with shrinkage method "glasso".
    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssx = np.column_stack(ssx)
    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()

    if whitening is not None:
        ssy = np.matmul(whitening, ssy).flatten()
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
        loglik = -math.inf

    return np.array([loglik])


def gaussian_syn_likelihood_ghurye_olkin(*ssx, observed=None, **kwargs):
    """Calculate the unbiased posterior logpdf.

    Uses the unbiased estimator of the synthetic likelihood.

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x
    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssx = np.column_stack(ssx)
    n, d = ssx.shape
    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()
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
        print('Matrix is not positive definite')
        loglik = -math.inf

    return np.array([loglik])


def semi_param_kernel_estimate(*ssx, shrinkage=None, penalty=None,
                               whitening=None, standardise=False,
                               observed=None, tkde=False):
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
    ssx :  Simulated summaries at x
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
        for heaving shrinkage to be applied (hence smaller batch_size).
    standardise : bool, optional
        Used with shrinkage method "glasso".

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssx = np.column_stack(ssx)

    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()

    n, ns = ssx.shape

    logpdf_y = np.zeros(ns)
    y_u = np.zeros(ns)
    sim_eta = np.zeros((n, ns))  # only used for wsemibsl
    eta_cov = None
    jacobian = 1  # used for TKDE method
    for j in range(ns):
        ssx_j = ssx[:, j].flatten()
        y = ssy[j]

        # NOTE: bw_method - "silverman" is being used here is slightly
        #       different than "nrd0" - silverman's rule of thumb in R.
        kernel = ss.kde.gaussian_kde(ssx_j, bw_method="silverman")
        logpdf_y[j] = kernel.logpdf(y) * np.abs(jacobian)

        y_u[j] = kernel.integrate_box_1d(np.NINF, y)

        if whitening is not None:
            # TODO? Commented out very inefficient for large batch_size
            # sim_eta[:, j] = [ss.norm.ppf(kernel.integrate_box_1d(np.NINF,
            #                                                      ssx_i))
            #                  for ssx_i in ssx_j]
            sim_eta[:, j] = ss.norm.ppf(ss.rankdata(ssx_j)/(n+1))

    # Below is exit point for helper function for estimate_whitening_matrix
    if not hasattr(whitening, 'shape') and whitening == "whitening":
        return sim_eta

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


def syn_likelihood_misspec(self, *ssx, adjustment="variance", tau=0.5,
                           penalty=None, whitening=None, observed=None,
                           w=1, **kwargs):
    """Calculate the posterior logpdf using the standard synthetic likelihood.

    Parameters
    ----------
    ssx :  Simulated summaries at x
    shrinkage : str, optional
        The shrinkage method to be used with the penalty param. With "glasso"
        this corresponds with BSLasso and with "warton" this corresponds
        with wBsl.
    adjustment : str
        String name of type of misspecified BSL. Can be either "mean" or
        "variance".
    tau : float, optional
        Scale (or inverse rate) parameter for the Laplace prior distribution of
        gamma. Defaults to 1.
    penalty : float, optional
        The penalty value to used for the specified shrinkage method.
        Must be between zero and one when using shrinkage method "Warton".
    whitening :  np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller batch_size).

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssx = np.column_stack(ssx)
    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()
    s1, s2 = ssx.shape
    dim_ss = len(ssy)

    batch_idx = kwargs['meta']['batch_index']
    prev_iter_loglik = self.state['prev_iter_logliks'][batch_idx]  # TODO -1?
    prev_std = self.state['stdevs'][batch_idx]
    prev_sample_mean = self.state['sample_means'][batch_idx]
    prev_sample_cov = self.state['sample_covs'][batch_idx]
    # first iter -> does not use mean/var - adjustment
    gamma = None
    if prev_iter_loglik is not None:
        gamma = self.state['gammas'][batch_idx-1]
        if gamma is None:
            if adjustment == "mean":
                gamma = np.repeat(0., dim_ss)
            if adjustment == "variance":
                gamma = np.repeat(tau, dim_ss)
        if adjustment == "mean":
            gamma, loglik = slice_gamma_mean(ssx,
                                             ssy=ssy,
                                             loglik=prev_iter_loglik,
                                             gamma=gamma,
                                             std=prev_std,
                                             sample_mean=prev_sample_mean,
                                             sample_cov=prev_sample_cov,
                                             tau=tau,
                                             w=w
                                             )
        if adjustment == "variance":
            gamma, loglik = slice_gamma_variance(ssx,
                                                 ssy=ssy,
                                                 loglik=prev_iter_loglik,
                                                 gamma=gamma,
                                                 std=prev_std,
                                                 sample_mean=prev_sample_mean,
                                                 sample_cov=prev_sample_cov,
                                                 tau=tau,
                                                 w=w
                                                 )

        self.update_gamma(gamma)
        self.update_slice_sampler_logliks(loglik)
    if s1 == dim_ss:  # obs as columns
        ssx = np.transpose(ssx)

    sample_mean = ssx.mean(0)
    sample_cov = np.cov(ssx, rowvar=False)

    std = np.std(ssx, axis=0)
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
