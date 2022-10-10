"""Example implementation of the Machand toad model.

This model simulates the movement of Fowler's toad species.
"""

import logging
import warnings
from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def toad(alpha,
         gamma,
         p0,
         n_toads=66,
         n_days=63,
         batch_size=1,
         random_state=None):
    """Sample the movement of Fowler's toad species.

    Models foraging steps using a levy_stable distribution where individuals
    either return to a previous site or establish a new one.

    References
    ----------
    Marchand, P., Boenke, M., and Green, D. M. (2017).
    A stochastic movement model reproduces patterns of site fidelity and long-
    distance dispersal in a population of fowlers toads (anaxyrus fowleri).
    Ecological Modelling,360:63–69.

    Parameters
    ----------
    alpha : float or array_like with batch_size entries
        step length distribution stability parameter
    gamma : float or array_like with batch_size entries
        step lentgh distribution scale parameter
    p0 : float or array_like with batch_size entries
        probability of returning to a previous refuge site
    n_toads : int, optional
        number of toads
    n_days : int, optional
        number of days
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
    np.ndarray in shape (n_days x n_toads x batch_size)

    """
    X = np.zeros((n_days, n_toads, batch_size))
    random_state = random_state or np.random
    step_gen = ss.levy_stable
    step_gen.random_state = random_state

    for i in range(1, n_days):
        ret = random_state.uniform(0, 1, (n_toads, batch_size)) < np.squeeze(p0)
        non_ret = np.invert(ret)

        delta_x = step_gen.rvs(alpha, beta=0, scale=gamma, size=(n_toads, batch_size))
        X[i, non_ret] = X[i-1, non_ret] + delta_x[non_ret]

        ind_refuge = random_state.choice(i, size=(n_toads, batch_size))
        X[i, ret] = X[ind_refuge[ret], ret]

    return X


def compute_summaries(X, lag, p=np.linspace(0, 1, 11), thd=10):
    """Compute summaries for toad model.

    For displacements over lag...
        Log of the differences in the p quantiles
        The number of absolute displacements less than thd
        Median of the absolute displacements greater than thd

    Parameters
    ----------
    X : np.array of shape (ndays x ntoads x batch_size)
        observed matrix of toad displacements
    lag : list of ints, optional
        the number of days behind to compute displacement with
    p : np.array, optional
        quantiles used in summary statistic calculation (default 0, 0.1, ... 1)
    thd : float
        toads are considered returned when absolute displacement does not exceed thd (default 10m)

    Returns
    -------
    np.ndarray in shape (batch_size x len(p) + 1)

    """
    disp = obs_mat_to_deltax(X, lag)  # num disp at lag x batch size
    abs_disp = np.abs(disp)
    # returned toads
    ret = abs_disp < thd
    num_ret = np.sum(ret, axis=0)
    # non-returned toads
    abs_disp[ret] = np.nan  # ignore returned toads
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        abs_noret_median = np.nanmedian(abs_disp, axis=0)
        abs_noret_quantiles = np.nanquantile(abs_disp, p, axis=0)
    diff = np.diff(abs_noret_quantiles, axis=0)
    logdiff = np.log(np.maximum(diff, np.exp(-20)))  # substitute zeros with a small positive
    # combine
    ssx = np.vstack((num_ret, abs_noret_median, logdiff))
    ssx = np.nan_to_num(ssx, nan=np.inf)  # nans are when all toads returned
    return np.transpose(ssx)


def obs_mat_to_deltax(X, lag):
    """Convert an observation matrix to a vector of displacements.

    Parameters
    ----------
    X : np.array (n_days x n_toads x batch_size)
        observed matrix of toad displacements
    lag : int
        the number of days behind to compute displacement with

    Returns
    -------
    np.ndarray in shape (n_toads * (n_days - lag) x batch_size)

    """
    batch_size = np.atleast_3d(X).shape[-1]
    return (X[lag:] - X[:-lag]).reshape(-1, batch_size)


def get_model(true_params=None, seed_obs=None):
    """Return a complete toad model in inference task.

    Parameters
    ----------
    true_params : list, optional
        parameters with which the observed data is generated
    seed_obs : int, optional
        seed for the observed data generation

    Returns
    -------
    m : elfi.ElfiModel

    """
    logger = logging.getLogger()
    if true_params is None:
        true_params = [1.7, 35.0, 0.6]

    m = elfi.ElfiModel()

    y = toad(*true_params, random_state=np.random.RandomState(seed_obs))

    elfi.Prior('uniform', 1, 1, model=m, name='alpha')
    elfi.Prior('uniform', 0, 100, model=m, name='gamma')
    elfi.Prior('uniform', 0, 0.9, model=m, name='p0')
    elfi.Simulator(toad, m['alpha'], m['gamma'], m['p0'], observed=y, name='toad')
    S1 = elfi.Summary(partial(compute_summaries, lag=1), m['toad'], name='S1')
    S2 = elfi.Summary(partial(compute_summaries, lag=2), m['toad'], name='S2')
    S4 = elfi.Summary(partial(compute_summaries, lag=4), m['toad'], name='S4')
    S8 = elfi.Summary(partial(compute_summaries, lag=8), m['toad'], name='S8')
    # NOTE: toad written for BSL, distance node included but not tested
    elfi.Distance('euclidean', S1, S2, S4, S8, name='d')

    logger.info("Generated observations with true parameters "
                "alpha: %.1f, gamma: %d, p0: %.1f, ", *true_params)

    return m
