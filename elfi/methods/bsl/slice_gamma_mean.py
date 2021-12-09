"""Slice sampler to find mean adjustment parameter values as specified in:
Robust Approximate Bayesian Inference With Synthetic Likelihood.
Journal of Computational and Graphical Statistics. 1-39.
10.1080/10618600.2021.1875839.
"""

import numpy as np
import math
import scipy.stats as ss


def log_gamma_prior(x, tau=1.0):
    """Laplace prior for gamma values

    Parameters
    ----------
    x : np.array
        Gamma values
    tau: int, optional

    Returns
    -------
    density at x
    """
    n = len(x)
    rate = 1/tau
    res = n * math.log(rate/2) - rate * np.sum(np.abs(x))
    return res


def slice_gamma_mean(ssx, ssy, loglik, gamma, std, sample_mean, sample_cov,
                     tau=1.0, w=1.0, max_iter=1000):
    """Slice sampler algorithm for mean adjustment gammas

    Parameters
    ----------
    ssx : np.array
        Simulated summaries
    loglik : np.float64
        Current log-likelihood
    gamma : np.array
        gamma of previous iteration
    sample_mean : np.array
        sample mean from previous iteration
    sample_cov : np.array
        sample cov from previous iteration
    tau : float, optional
        Scale (or inverse rate) parameter of the Laplace prior
        distribution for gamma.
    w : float, optional
        Step size used for stepping out procedure in slice sampler.
    max_iter : int, optional
        The maximum number of iterations for the stepping out and shrinking
        procedures for the slice sampler algorithm.
    Returns
    -------
    gamma_curr : np.array
    """
    gamma_curr = gamma
    for ii, gamma in enumerate(gamma_curr):
        target = loglik + log_gamma_prior(gamma_curr) - \
                np.random.exponential(1)  # TODO? -> need random seed

        lower = gamma - w
        upper = gamma + w

        # stepping out procedure for lower bound
        i = 0
        while (i <= max_iter):
            gamma_lower = gamma_curr
            gamma_lower[ii] = lower
            mu_lower = sample_mean + std * gamma_lower
            loglik = ss.multivariate_normal.logpdf(
                ssy,
                mean=mu_lower,
                cov=sample_cov
                )
            prior = log_gamma_prior(gamma_lower)
            target_lower = loglik + prior
            if target_lower < target:
                break
            lower = lower - 1
            i += 1

        # stepping out procedure for upper bound
        i = 0
        while (i <= max_iter):
            gamma_upper = gamma_curr
            gamma_upper[ii] = upper
            mu_upper = sample_mean + std * gamma_upper
            loglik = ss.multivariate_normal.logpdf(
                ssy,
                mean=mu_upper,
                cov=sample_cov
                )
            prior = log_gamma_prior(gamma_upper)
            target_upper = loglik + prior
            if target_upper < target:
                break
            upper = upper + 1
            i += 1

        # shrink
        i = 0
        while (i < max_iter):
            prop = np.random.uniform(lower, upper)  # TODO? -> need random seed
            gamma_prop = gamma_curr
            gamma_prop[ii] = prop
            sample_mean_prop = sample_mean + std * gamma_prop
            loglik = ss.multivariate_normal.logpdf(
                ssy,
                mean=sample_mean_prop,
                cov=sample_cov
                )
            prior = log_gamma_prior(gamma_prop)
            target_prop = loglik + prior

            if target_prop > target:
                gamma_curr = gamma_prop
                break
            if prop < gamma:
                lower = prop
            else:
                upper = prop
            i += 1

    return gamma_curr
