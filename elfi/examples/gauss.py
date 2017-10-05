"""Implementations of Gaussian noise example models."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


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
    array_like
        1-D observations.

    """
    # Transforming the arrays' shape to be compatible with batching.
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
    array_like
        n-D observations.

    """
    n_dim = len(mu)

    # Transforming the arrays' shape to be compatible with batching.
    batches_mu = np.zeros(shape=(batch_size, n_dim))
    for idx_dim, param_mu in enumerate(mu):
        batches_mu[:, idx_dim] = param_mu

    # Sampling the observations.
    y_obs = np.zeros(shape=(batch_size, n_obs, n_dim))
    for idx_batch in range(batch_size):
        y_batch = ss.multivariate_normal.rvs(mean=batches_mu[idx_batch], cov=cov_matrix,
                                             size=n_obs, random_state=random_state)
        if n_dim == 1:
            y_batch = y_batch[:, np.newaxis]
        y_obs[idx_batch, :, :] = y_batch
    return y_obs


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
    cov_matrix : array_like, optional
        Covariance matrix, a requirement for the nd_mean model.

    Returns
    -------
    elfi.ElfiModel

    """
    # Defining the default settings.
    if true_params is None:
        if nd_mean:
            true_params = [4, 4]  # 2-D mean.
        else:
            true_params = [4, .4]  # mean and standard deviation.

    # Choosing the simulator for both observations and simulations.
    if nd_mean:
        fn_simulator = partial(gauss_nd_mean, cov_matrix=cov_matrix, n_obs=n_obs)
    else:
        fn_simulator = partial(gauss, n_obs=n_obs)

    # Obtaining the observations.
    y_obs = fn_simulator(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))

    m = elfi.new_model()
    # Initialising the priors.
    eps_prior = 5  # The longest distance from the median of an initialised prior's distribution.
    priors = []
    if nd_mean:
        n_dim = len(true_params)
        for i in range(n_dim):
            name_prior = 'mu_{}'.format(i)
            prior_mu = elfi.Prior('uniform', true_params[i] - eps_prior,
                                  2 * eps_prior, model=m, name=name_prior)
            priors.append(prior_mu)
    else:
        priors.append(elfi.Prior('uniform', true_params[0] - eps_prior,
                                 2 * eps_prior, model=m, name='mu'))
        priors.append(elfi.Prior('truncnorm', np.amax([.01, true_params[1] - eps_prior]),
                                 2 * eps_prior, model=m, name='sigma'))
    elfi.Simulator(fn_simulator, *priors, observed=y_obs, name='gauss')

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


def ss_mean(y):
    """Obtain the mean summary statistic.

    Parameters
    ----------
    y : array_like
        Yielded points.

    Returns
    -------
    array_like of the shape (batch_size, dim_point)

    """
    ss = np.mean(y, axis=1)
    return ss


def ss_var(y):
    """Return the variance summary statistic.

    Parameters
    ----------
    y : array_like
        Yielded points.

    Returns
    -------
    array_like of the shape (batch_size, dim_point)

    """
    ss = np.var(y, axis=1)
    return ss


def euclidean_multidim(*simulated, observed):
    """Calculate the Euclidean distances merging data dimensions.

    The shape of the input arrays corresponds to (batch_size, dim_point).

    Parameters
    ----------
    *simulated: array_like
    observed : array_like

    Returns
    -------
    array_like

    """
    pts_sim = simulated[0]
    pts_obs = observed[0]

    # Integrating over the summary statistics.
    d_dim_merged = np.sum((pts_sim - pts_obs)**2., axis=1)

    d = np.sqrt(d_dim_merged)
    return d
