"""Example implementation of the Ricker model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def ricker(log_rate, stock_init=1., n_obs=50, batch_size=1, random_state=None):
    """Generate samples from the Ricker model.

    Ricker, W. E. (1954) Stock and Recruitment Journal of the Fisheries
    Research Board of Canada, 11(5): 559-623.

    Parameters
    ----------
    log_rate : float or np.array
        Log growth rate of population.
    stock_init : float or np.array, optional
        Initial stock.
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    stock : np.array

    """
    random_state = random_state or np.random

    stock = np.empty((batch_size, n_obs))
    stock[:, 0] = stock_init

    for ii in range(1, n_obs):
        stock[:, ii] = stock[:, ii - 1] * np.exp(log_rate - stock[:, ii - 1])

    return stock


def stochastic_ricker(log_rate,
                      std,
                      scale,
                      stock_init=1.,
                      n_obs=50,
                      batch_size=1,
                      random_state=None):
    """Generate samples from the stochastic Ricker model.

    Here the observed stock ~ Poisson(true stock * scaling).

    Parameters
    ----------
    log_rate : float or np.array
        Log growth rate of population.
    std : float or np.array
        Standard deviation of innovations.
    scale : float or np.array
        Scaling of the expected value from Poisson distribution.
    stock_init : float or np.array, optional
        Initial stock.
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    stock_obs : np.array

    """
    random_state = random_state or np.random

    stock_obs = np.empty((batch_size, n_obs))
    stock_prev = stock_init

    for ii in range(n_obs):
        stock = stock_prev * np.exp(log_rate - stock_prev + std * random_state.randn(batch_size))
        stock_prev = stock

        # the observed stock is Poisson distributed
        stock_obs[:, ii] = random_state.poisson(scale * stock, batch_size)

    return stock_obs


def get_model(n_obs=50, true_params=None, seed_obs=None, stochastic=True):
    """Return a complete Ricker model in inference task.

    This is a simplified example that achieves reasonable predictions. For more extensive treatment
    and description using 13 summary statistics, see:

    Wood, S. N. (2010) Statistical inference for noisy nonlinear ecological dynamic systems,
    Nature 466, 1102–1107.

    Parameters
    ----------
    n_obs : int, optional
        Number of observations.
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.
    stochastic : bool, optional
        Whether to use the stochastic or deterministic Ricker model.

    Returns
    -------
    m : elfi.ElfiModel

    """
    if stochastic:
        simulator = partial(stochastic_ricker, n_obs=n_obs)
        if true_params is None:
            true_params = [3.8, 0.3, 10.]

    else:
        simulator = partial(ricker, n_obs=n_obs)
        if true_params is None:
            true_params = [3.8]

    m = elfi.ElfiModel()
    y_obs = simulator(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(simulator, n_obs=n_obs)
    sumstats = []

    if stochastic:
        elfi.Prior(ss.expon, np.e, 2, model=m, name='t1')
        elfi.Prior(ss.truncnorm, 0, 5, model=m, name='t2')
        elfi.Prior(ss.uniform, 0, 100, model=m, name='t3')
        elfi.Simulator(sim_fn, m['t1'], m['t2'], m['t3'], observed=y_obs, name='Ricker')
        sumstats.append(elfi.Summary(partial(np.mean, axis=1), m['Ricker'], name='Mean'))
        sumstats.append(elfi.Summary(partial(np.var, axis=1), m['Ricker'], name='Var'))
        sumstats.append(elfi.Summary(num_zeros, m['Ricker'], name='#0'))
        elfi.Discrepancy(chi_squared, *sumstats, name='d')

    else:  # very simple deterministic case
        elfi.Prior(ss.expon, np.e, model=m, name='t1')
        elfi.Simulator(sim_fn, m['t1'], observed=y_obs, name='Ricker')
        sumstats.append(elfi.Summary(partial(np.mean, axis=1), m['Ricker'], name='Mean'))
        elfi.Distance('euclidean', *sumstats, name='d')

    return m


def chi_squared(*simulated, observed):
    """Return Chi squared goodness of fit.

    Adjusts for differences in magnitude between dimensions.

    Parameters
    ----------
    simulated : np.arrays
    observed : tuple of np.arrays

    """
    simulated = np.column_stack(simulated)
    observed = np.column_stack(observed)
    d = np.sum((simulated - observed)**2. / observed, axis=1)
    return d


def num_zeros(x):
    """Return a summary statistic: number of zero observations."""
    n = np.sum(x == 0, axis=1)
    return n
