"""Example implementation of the M/G/1 Queue model.

References
----------
An et al (2020) Robust Bayesian synthetic likelihood via a semi-parametric approach.
Stat Comput, 30: 543-557. https://doi.org/10.1007/s11222-019-09904-x

Blum and Francois (2010) Non-linear regression models for Approximate Bayesian
Computation. Stat Comput, 20: 63-73. https://doi.org/10.1007/s11222-009-9116-0

"""

import logging
from functools import partial

import numpy as np

import elfi


def MG1(t1, t2, t3, n_obs=50, batch_size=1, random_state=None):
    """Generate a sequence of samples from the M/G/1 model.

    Parameters
    ----------
    t1 : float, array_like
        minimum service time length
    t2 : float, array_like
        maximum service time length
    t3 : float, array_like
        Time between arrivals Exp(t3) distributed
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    """
    random_state = random_state or np.random

    # arrival time of customer j after customer j - 1
    W = random_state.exponential(1/t3, size=(n_obs, batch_size))    # beta = 1/lmda
    # service times
    U = random_state.uniform(t1, t2, size=(n_obs, batch_size))

    y = np.zeros((n_obs, batch_size))
    sum_w = np.zeros(batch_size)
    sum_x = np.zeros(batch_size)

    for i in range(n_obs):
        sum_w += W[i]    # i-th arrival time = previous arrival + i-th interarrival time
        y[i] = U[i] + np.maximum(0, sum_w - sum_x)
        sum_x += y[i]    # i-th departure time = previous departure + i-th interdeparture time

    return np.transpose(y)


def log_identity(x):
    """Return log observations as summary."""
    return np.log(x)


def quantiles(x, q):
    """Return selected quantiles as summary."""
    qs = np.quantile(x, q, axis=1)
    return np.transpose(qs)


def get_model(n_obs=50, true_params=None, seed_obs=None, n_quantiles=10):
    """Return a complete M/G/1 model in inference task.

    Parameters
    ----------
    n_obs : int, optional
        observation length of the MA2 process
    true_params : list, optional
        parameters with which the observed data is generated
    seed_obs : int, optional
        seed for the observed data generation
    n_quantiles : int, optional
        number of equidistant quantiles to be used as summary statistics

    Returns
    -------
    m : elfi.ElfiModel

    """
    logger = logging.getLogger()
    if true_params is None:
        true_params = [1., 5., 0.2]

    y = MG1(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(MG1, n_obs=n_obs)

    # TODO: CHECK CONSTRAINT LOGIC
    # constraint_t1, constraint_t2 = theta_constraints(y)

    m = elfi.ElfiModel()
    elfi.Prior('uniform', 0, 10, model=m, name='t1')
    elfi.Prior('uniform', m['t1'], 10, model=m, name='t2')  # t2-t1 ~ U(0,10)
    elfi.Prior('uniform', 0, 0.5, model=m, name='t3')

    elfi.Simulator(sim_fn, m['t1'], m['t2'], m['t3'], observed=y, name='MG1')

    # log interdeparture times (An et al, 2020)
    elfi.Summary(log_identity, m['MG1'], name='log_identity')

    # equidistant quantiles (Blum and Francois, 2010)
    q = np.linspace(0, 1, n_quantiles)
    elfi.Summary(partial(quantiles, q=q), m['MG1'], name='quantiles')

    # NOTE: M/G/1 written for BSL, distance node included but not well tested
    elfi.Distance('euclidean', m['quantiles'], w=(1/100)**q, name='d')

    logger.info("Generated observations with true parameters "
                "t1: %.1f, t2: %.1f, t3: %.1f, ", *true_params)

    return m
