"""Example implementation of the M/G/1 Queue model."""

from functools import partial

import numpy as np

import elfi


def MG1(t1, t2, t3, n_obs=50, batch_size=1, random_state=None):
    """Generate a sequence of samples from the M/G/1 model.

    The sequence is a moving average

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
    if hasattr(t1, 'shape'):  # assumes vector consists of identical values
        t1, t2, t3 = t1[0], t2[0], t3[0]

    random_state = random_state or np.random

    # arrival time of customer j after customer j - 1
    W = random_state.exponential(1/t3, size=(batch_size, n_obs))  # beta = 1/lmda
    # service times
    U = random_state.uniform(t1, t2, size=(batch_size, n_obs))

    y = np.zeros((batch_size, n_obs))
    sum_w = W[:, 0]  # arrival time of jth customer, init first time point
    sum_x = np.zeros(batch_size)  # departure time of the prev customer, init 0s

    for i in range(n_obs):
        y[:, i] = U[:, i].flatten() + np.maximum(np.zeros(batch_size), sum_w - sum_x).flatten()
        sum_w += W[:, i]
        sum_x += y[:, i-1]

    return y


def log_identity(x):
    """Return log observations as summary."""
    return np.log(x)


def identity(x):
    """Return observations as summary."""
    return x


def get_model(n_obs=50, true_params=None, seed_obs=None):
    """Return a complete M/G/1 model in inference task.

    Parameters
    ----------
    n_obs : int, optional
        observation length of the MA2 process
    true_params : list, optional
        parameters with which the observed data is generated
    seed_obs : int, optional
        seed for the observed data generation

    Returns
    -------
    m : elfi.ElfiModel

    """
    if true_params is None:
        true_params = [1., 5., 0.2]

    y = MG1(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(MG1, n_obs=n_obs)

    # TODO: CHECK CONSTRAINT LOGIC
    # constraint_t1, constraint_t2 = theta_constraints(y)

    m = elfi.ElfiModel()
    elfi.Prior('uniform', 0, np.min(y), model=m, name='t1')
    elfi.Prior('uniform', m['t1'], 10, model=m, name='t2')  # t2-t1 ~ U(0,10)
    elfi.Prior('uniform', 0, 0.5, model=m, name='t3')

    elfi.Simulator(sim_fn, m['t1'], m['t2'], m['t3'], observed=y, name='MG1')

    elfi.Summary(log_identity, m['MG1'], name='log_identity')

    # NOTE: M/G/1 written for BSL, distance node included but not well tested
    elfi.Distance('euclidean', m['log_identity'], name='d')

    elfi.SyntheticLikelihood("bsl", m['log_identity'], name="SL")

    return m
