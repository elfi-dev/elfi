"""The general, n-dimensional Gaussian noise model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi
from elfi.examples.gnk import euclidean_multidim


def Gauss_nd(*params, cov_ii=None, cov_ij=None, n_obs=15, batch_size=1, random_state=None):
    """Sample the Gaussian distribution.

    Reference
    ---------
    The default settings replicate the experiment settings used in [1].

    [1] arXiv:1704.00520 (Järvenpää et al., 2017).

    Parameters
    ----------
    *params : array_like
        The array elements correspond to the mean parameters.
    cov_ii : float, optional
        The diagonal variance.
    cov_ij : float, optional
        The non-diagonal variance.
    n_obs : int, optional
        The number of observations.
    batch_size : int, optional
        The number of batches.
    random_state : np.random.RandomState, optional

    Returns
    -------
    y : array_like

    """
    n_dim = len(params)
    # Formatting the mean.
    mu = np.zeros(shape=(batch_size, n_dim))
    for idx_dim, param_mu in enumerate(params):
        mu[:, idx_dim] = param_mu
    # Formatting the diagonal covariance.
    cov_ii = np.repeat(cov_ii, batch_size)
    if batch_size == 1:
        cov_ii = cov_ii[None]
    # Formatting the non-diagonal covariance.
    if n_dim != 1:
        cov_ij = np.repeat(cov_ij, batch_size)
        if batch_size == 1:
            cov_ij = cov_ij[None]
    # Creating the covariance matrix.
    cov = np.zeros(shape=(batch_size, n_dim, n_dim))
    for idx_batch in range(batch_size):
        if n_dim != 1:
            cov[idx_batch].fill(np.asscalar(cov_ij[idx_batch]))
        np.fill_diagonal(cov[idx_batch], cov_ii[idx_batch])
    # Sampling observations.
    y = np.zeros(shape=(batch_size, n_obs, n_dim))
    for idx_batch in range(batch_size):
        y_batch = ss.multivariate_normal.rvs(mean=mu[idx_batch],
                                             cov=cov[idx_batch],
                                             size=n_obs,
                                             random_state=random_state)
        if n_dim == 1:
            y_batch = y_batch[:, None]
        y[idx_batch, :, :] = y_batch
    return y


def ss_mean(x):
    """Return the summary statistic corresponding to the mean."""
    ss = np.mean(x, axis=1)
    return ss


def ss_var(x):
    """Return the summary statistic corresponding to the variance."""
    ss = np.var(x, axis=1)
    return ss


def get_model(true_params=None, cov_ii=1, cov_ij=.5, n_obs=15, seed_obs=None):
    """Return an initialised Gaussian noise model.

    Parameters
    ----------
    true_params : array_like
        The array elements correspond to the mean parameters.
    cov_ii : float, optional
        The diagonal variance.
    cov_ij : float, optional
        The non-diagonal variance.
    n_obs : int, optional
        The number of observations.
    random_state : np.random.RandomState, optional

    Returns
    -------
    m : elfi.ElfiModel

    """
    # The default settings use the 2-D Gaussian model.
    if true_params is None:
        true_params = [4, 4]
    n_dim = len(true_params)
    # Obtaining the observations.
    y_obs = Gauss_nd(*true_params,
                     cov_ii=cov_ii,
                     cov_ij=cov_ij,
                     n_obs=n_obs,
                     random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(
        Gauss_nd, cov_ii=cov_ii, cov_ij=cov_ij, n_obs=n_obs)
    m = elfi.ElfiModel()
    # Defining the priors.
    priors = []
    for i in range(n_dim):
        name_prior = 'mu_{}'.format(i)
        prior_mu = elfi.Prior('uniform', 0, 8, model=m, name=name_prior)
        priors.append(prior_mu)

    elfi.Simulator(sim_fn, *priors, observed=y_obs, name='gm')
    elfi.Summary(ss_mean, m['gm'], name='ss_mean')
    elfi.Summary(ss_var, m['gm'], name='ss_var')
    elfi.Discrepancy(euclidean_multidim, m['ss_mean'], m['ss_var'], name='d')

    return m
