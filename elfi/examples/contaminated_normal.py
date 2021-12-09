"""Example implementation of the contaminated normal model"""

from functools import partial

import numpy as np

import elfi


def contaminated_normal(theta, w=0.8, n_obs=100, batch_size=1, sigma_eps=2.5, random_state=None):
    random_state = random_state or np.random
    # if len(theta) != n_obs:
    #     theta = np.repeat(theta, n_obs)
    means = np.repeat(theta, n_obs)
    sigmas = random_state.choice([1, sigma_eps], size=n_obs*batch_size, p=[w, 1-w])
    x = random_state.normal(means, sigmas)
    x = x.reshape((batch_size, n_obs))
    return x


def get_model(n_obs=100, true_params=None, seed_obs=None):
    """Returns a complete contaminated normal model in inference task.

    Args:
        n_obs (int, optional): [description]. Defaults to 100.
        true_params ([type], optional): [description]. Defaults to None.
        seed_obs ([type], optional): [description]. Defaults to None.
    """
    if true_params is None:
        true_params = [1]

    sim_fn = partial(contaminated_normal, n_obs=n_obs)
    # TODO: handle observed
    m = elfi.ElfiModel()
    elfi.Prior('normal', 0, 10, model=m, name='theta')
    elfi.Simulator(sim_fn, m['theta'], name='contaminated_normal')
    # TODO: check mean, var axis
    elfi.Summary(partial(np.mean, axis=1), m['contaminated_normal'], name='Mean')
    elfi.Summary(partial(np.var, axis=1), m['contaminated_normal'], name='Var')

    return m
