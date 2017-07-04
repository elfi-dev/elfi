"""An example implementation of a Gaussian noise model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def Gauss(mu, sigma, n_obs=50, batch_size=1, random_state=None):
    """Sample the Gaussian distribution.

    Parameters
    ----------
    mu : float, array_like
    sigma : float, array_like
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    """
    # Standardising the parameter's format.
    mu = np.asanyarray(mu).reshape((-1, 1))
    sigma = np.asanyarray(sigma).reshape((-1, 1))
    y = ss.norm.rvs(loc=mu, scale=sigma, size=(batch_size, n_obs), random_state=random_state)
    return y


def ss_mean(x):
    """Return the summary statistic corresponding to the mean."""
    ss = np.mean(x, axis=1)
    return ss


def ss_var(x):
    """Return the summary statistic corresponding to the variance."""
    ss = np.var(x, axis=1)
    return ss


def get_model(n_obs=50, true_params=None, seed_obs=None):
    """Return a complete Gaussian noise model.

    Parameters
    ----------
    n_obs : int, optional
        the number of observations
    true_params : list, optional
        true_params[0] corresponds to the mean,
        true_params[1] corresponds to the standard deviation
    seed_obs : int, optional
        seed for the observed data generation

    Returns
    -------
    m : elfi.ElfiModel

    """
    if true_params is None:
        true_params = [1, 1]

    y_obs = Gauss(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(Gauss, n_obs=n_obs)

    m = elfi.ElfiModel()
    elfi.Prior('uniform', -2, 4, model=m, name='mu')
    elfi.Prior('truncnorm', 0.01, 5, model=m, name='sigma')
    elfi.Simulator(sim_fn, m['mu'], m['sigma'], observed=y_obs, name='gm')
    elfi.Summary(ss_mean, m['gm'], name='ss_mean')
    elfi.Summary(ss_var, m['gm'], name='ss_var')
    elfi.Distance('euclidean', m['ss_mean'], m['ss_var'], name='d')

    return m
