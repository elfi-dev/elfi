"""Example implementations of Gaussian noise models."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi
from elfi.examples.gnk import euclidean_multidim


def gauss(mu, sigma, n_obs=50, batch_size=1, random_state=None):
    """Sample the 1-D Gaussian distribution.

    Parameters
    ----------
    mu : float, array_like
    sigma : float, array_like
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    y_obs : array_like
        1-D observation.

    """
    # Handling batching.
    batches_mu = np.asanyarray(mu).reshape((-1, 1))
    batches_sigma = np.asanyarray(sigma).reshape((-1, 1))

    # Sampling observations.
    y_obs = ss.norm.rvs(loc=batches_mu, scale=batches_sigma,
                        size=(batch_size, n_obs), random_state=random_state)
    return y_obs


def gauss_nd_mean(*mu, cov_matrix, n_obs=15, batch_size=1, random_state=None):
    """Sample an n-D Gaussian distribution.

    Parameters
    ----------
    *mu : array_like
        Mean parameters.
    cov_matrix : array_like
        Covariance matrix.
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    y_obs : array_like
        n-D observation.

    """
    n_dim = len(mu)

    # Handling batching.
    batches_mu = np.zeros(shape=(batch_size, n_dim))
    for idx_dim, param_mu in enumerate(mu):
        batches_mu[:, idx_dim] = param_mu

    # Sampling the observations.
    y_obs = np.zeros(shape=(batch_size, n_obs, n_dim))
    for idx_batch in range(batch_size):
        y_batch = ss.multivariate_normal.rvs(mean=batches_mu[idx_batch],
                                             cov=cov_matrix,
                                             size=n_obs,
                                             random_state=random_state)
        if n_dim == 1:
            y_batch = y_batch[:, np.newaxis]
        y_obs[idx_batch, :, :] = y_batch
    return y_obs


def ss_mean(x):
    """Return the summary statistic corresponding to the mean."""
    ss = np.mean(x, axis=1)
    return ss


def ss_var(x):
    """Return the summary statistic corresponding to the variance."""
    ss = np.var(x, axis=1)
    return ss


def get_model(n_obs=50, true_params=None, seed_obs=None, nd_mean=False, cov_matrix=None):
    """Return a Gaussian noise model.

    Parameters
    ----------
    n_obs : int, optional
    true_params : list, optional
        Default parameter settings.
    seed_obs : int, optional
        Seed for the observed data generation.
    nd_mean : bool, optional
        Option to use an n-D mean Gaussian noise model.
    cov_matrix : None, optional
        Covariance matrix, a requirement for the nd_mean model.

    Returns
    -------
    m : elfi.ElfiModel

    """
    # Defining the default settings.
    if true_params is None:
        if nd_mean:
            true_params = [4, 4]  # 2-D mean.
        else:
            true_params = [4, .4]  # mean and standard deviation.

    # Choosing the simulator for both observations and simulations.
    if nd_mean:
        sim_fn = partial(gauss_nd_mean, cov_matrix=cov_matrix, n_obs=n_obs)
    else:
        sim_fn = partial(gauss, n_obs=n_obs)

    # Obtaining the observations.
    y_obs = sim_fn(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))

    m = elfi.ElfiModel()
    # Initialising the priors.
    priors = []
    if nd_mean:
        n_dim = len(true_params)
        for i in range(n_dim):
            name_prior = 'mu_{}'.format(i)
            prior_mu = elfi.Prior('uniform', 0, 8, model=m, name=name_prior)
            priors.append(prior_mu)
    else:
        priors.append(elfi.Prior('uniform', 0, 8, model=m, name='mu'))
        priors.append(elfi.Prior('truncnorm', 0.01, 5, model=m, name='sigma'))
    elfi.Simulator(sim_fn, *priors, observed=y_obs, name='gauss')

    # Initialising the summary statistics.
    sumstats = []
    sumstats.append(elfi.Summary(ss_mean, m['gauss'], name='ss_mean'))
    sumstats.append(elfi.Summary(ss_var, m['gauss'], name='ss_var'))

    # Choosing the discrepancy metric.
    if nd_mean:
        elfi.Discrepancy(euclidean_multidim, *sumstats, name='d')
    else:
        elfi.Distance('euclidean', *sumstats, name='d')

    return m
