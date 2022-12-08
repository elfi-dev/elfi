"""Example implementation of the AR1 model."""

import logging
from functools import partial

import numpy as np

import elfi


def AR1(phi, n_obs=200, batch_size=1, random_state=None):
    r"""Generate a sequence of samples from the AR1 model.

    The sequence is an autoregressive model

        x_i = \phi x_{i-1} + w_{i}

    where w_i are white noise ~ N(0,1) and x_0 = 0.

    Parameters
    ----------
    phi : float, array_like
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    """
    phi = np.asanyarray(phi)
    random_state = random_state or np.random

    # i.i.d. sequence ~ N(0,1)
    w = random_state.randn(batch_size, n_obs + 1)
    x = np.zeros((batch_size, n_obs+1))
    x_prev = np.zeros(batch_size)
    for i in range(1, n_obs+1):
        x[:, i] = phi * x_prev + w[:, i]
        x_prev = x[:, i]
    return x[:, 1:]


def get_model(n_obs=200, true_params=None, seed_obs=None):
    """Return a complete AR1 model in inference task.

    Parameters
    ----------
    n_obs : int, optional
        observation length of the MA2 process
    true_params : list, optional
        parameters with which the observed data is generated
    seed_obs : int, optional
        seed for the observed data generation    Returns
    -------
    m : elfi.ElfiModel

    """
    logger = logging.getLogger()
    if true_params is None:
        true_params = [.9]

    y = AR1(*true_params,
            n_obs=n_obs,
            random_state=np.random.RandomState(seed_obs))

    sim_fn = partial(AR1, n_obs=n_obs)

    m = elfi.ElfiModel()

    elfi.Prior('uniform', -1, 2, model=m, name='phi')
    elfi.Simulator(sim_fn, m['phi'], observed=y, name='AR1')

    elfi.Distance('euclidean', m['AR1'], name='d')

    logger.info("Generated observations with true parameter phi: %.1f.", *true_params)

    return m
