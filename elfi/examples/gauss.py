"""The general, n-dimensional Gaussian noise model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi
from elfi.examples.gnk import euclidean_multidim


def Gauss(*params, cov_ii=None, cov_ij=None, n_obs=15, batch_size=1, n_dim, random_state=None):
    """Sample the Gaussian distribution.

    Reference
    ---------
    The default settings replicate the experiment settings used by:
        [1] arXiv:1704.00520 (Järvenpää et al., 2017).

    Parameters
    ----------
    *params : array_like
        The first (n - 2) elements corresponds to the mean parameters,
        and the last two parameters correspond to the variance parameters.
    cov_ii : None, optional
        The diagonal variance.
    cov_ij : None, optional
        The non-diagonal variance.
    n_obs : int, optional
        The number of observations.
    batch_size : int, optional
        The number of batches.
    n_dim : int
        The number of dimensions.
    random_state : np.random.RandomState, optional

    Returns
    -------
    y : array_like

    Raises
    ------
    ValueError
        The number of dimension has to match the simulated parameters.
        A non-trivial check as the variance can be chosen not to be simulated.

    """
    if ((len(params) == n_dim) and
            (cov_ii is None or (cov_ij is None and n_dim != 1))):
        raise ValueError("Invalid dimension or parameters.")
    # Formatting the mean.
    mu = np.zeros(shape=(batch_size, n_dim))
    for idx_dim, param_mu in enumerate(params[:n_dim]):
        mu[:, idx_dim] = param_mu
    # Formatting the diagonal covariance.
    if cov_ii is None:
        cov_ii = params[n_dim]
        cov_ii = np.array(cov_ii)
    else:
        cov_ii = np.repeat(cov_ii, batch_size)
    if batch_size == 1:
        cov_ii = cov_ii[None]
    # Formatting the non-diagonal covariance.
    if n_dim != 1:
        if cov_ij is None:
            cov_ij = params[n_dim + 1]
            cov_ij = np.array(cov_ij)
        else:
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


def get_model(true_params=None, cov_ii=None, cov_ij=None, n_obs=15, n_dim=None, seed_obs=None):
    """Return an initialised Gaussian noise model.

    Parameters
    ----------
    true_params : array_like
        The first (n - 2) elements corresponds to the mean parameters,
        and the last two parameters correspond to the variance parameters.
    cov_ii : None, optional
        The diagonal variance.
    cov_ij : None, optional
        The non-diagonal variance.
    n_obs : int, optional
        The number of observations.
    n_dim : int, optional
        The number of dimensions.
    random_state : np.random.RandomState, optional

    Returns
    -------
    m : elfi.ElfiModel

    Raises
    ------
    ValueError
        The number of dimension has to match the simulated parameters.
        A non-trivial check as the variance can be chosen not to be simulated.

    """
    # The default settings use the 2-D Gaussian model:
    # - The first two elements is the 2-D mean;
    # - The third element is the diagonal (ii) variance;
    # - The fourth element is the non-diagonal (ij) variance.
    if true_params is None:
        true_params = [4, 4, 1, .5]
        n_dim = 2
    elif n_dim is None:
        n_dim = len(true_params)
    if ((len(true_params) == n_dim) and
            (cov_ii is None or (cov_ij is None and n_dim != 1))):
        raise ValueError("Invalid dimension or parameters.")
    y_obs = Gauss(*true_params,
                  cov_ii=cov_ii,
                  cov_ij=cov_ij,
                  n_obs=n_obs,
                  n_dim=n_dim,
                  random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(
        Gauss, cov_ii=cov_ii, cov_ij=cov_ij, n_obs=n_obs, n_dim=n_dim)
    m = elfi.ElfiModel()
    # Defining the priors.
    priors = []
    for i in range(n_dim):
        name_prior = 'mu_{}'.format(i)
        prior_mu = elfi.Prior('uniform', 0, 8, model=m, name=name_prior)
        priors.append(prior_mu)
    # Simulate the variance only if it was not pre-set.
    if cov_ii is None:
        prior_cov_ii = elfi.Prior('truncnorm', 0.01, 5, model=m, name='cov_ii')
        priors.append(prior_cov_ii)
    if cov_ij is None and n_dim != 1:
        prior_cov_ij = elfi.Prior('truncnorm', 0.01, 5, model=m, name='cov_ij')
        priors.append(prior_cov_ij)

    elfi.Simulator(sim_fn, *priors, observed=y_obs, name='gm')
    elfi.Summary(ss_mean, m['gm'], name='ss_mean')
    elfi.Summary(ss_var, m['gm'], name='ss_var')
    elfi.Discrepancy(euclidean_multidim, m['ss_mean'], m['ss_var'], name='d')

    return m
