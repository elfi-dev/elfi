"""Slice sampler to find variance adjustment parameter values.

Specified in:
Robust Approximate Bayesian Inference With Synthetic Likelihood.
Journal of Computational and Graphical Statistics. 1-39.
10.1080/10618600.2021.1875839.
"""


import numpy as np
import scipy.stats as ss


def log_gamma_prior(x, tau=1.0):
    """Exponential prior for gamma values.

    Parameters
    ----------
    x : np.array
        Gamma values

    Returns
    -------
    density at x

    """
    return np.sum(ss.expon.logpdf(x, scale=tau))  # tau - inv rate param, scale inv of rate.


def slice_gamma_variance(ssx, ssy, loglik, gamma, std, sample_mean, sample_cov,
                         tau=1.0, w=1.0, max_iter=1000, random_state=None):
    """Slice sampler algorithm for variance adjustment gammas.

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
    random_state = random_state or np.random
    gamma_curr = gamma
    for ii, gamma in enumerate(gamma_curr):
        target = loglik + log_gamma_prior(gamma_curr, tau) - \
            random_state.exponential(1)  # TODO? -> need random seed

        lower = 0
        upper = gamma + w

        # stepping out procedure for upper bound
        i = 0
        while (i <= max_iter):
            gamma_upper = gamma_curr
            gamma_upper[ii] = upper
            sample_cov_upper = sample_cov + np.diag((std * gamma_upper) ** 2)
            loglik = ss.multivariate_normal.logpdf(
                ssy,
                mean=sample_mean,
                cov=sample_cov_upper
                )
            prior = log_gamma_prior(gamma_upper, tau)
            target_upper = loglik + prior
            if target_upper < target:
                break
            upper = upper + 1
            i += 1

        # shrink
        i = 0
        while (i < max_iter):
            prop = np.random.uniform(lower, upper)  # TODO? -> need random seed
            gamma_prop = np.array(gamma_curr, dtype="float64")
            gamma_prop[ii] = prop
            sample_cov_upper = sample_cov + np.diag((std * gamma_prop) ** 2)
            loglik = ss.multivariate_normal.logpdf(
                ssy,
                mean=sample_mean,
                cov=sample_cov_upper
                )
            prior = log_gamma_prior(gamma_prop, tau)
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
