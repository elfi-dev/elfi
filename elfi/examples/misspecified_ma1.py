"""Example of inference for a SVM model misspecified as MA(1).

Approach follows:
Frazier, David & Drovandi, Christopher. (2021).
Robust Approximate Bayesian Inference With Synthetic Likelihood.
Journal of Computational and Graphical Statistics. 1-39.
10.1080/10618600.2021.1875839.
"""

from functools import partial

import numpy as np

import elfi


def MA1(t1, n_obs=100, batch_size=1, random_state=None):
    r"""Generate a sequence of samples from the MA2 model.

    The sequence is a moving average

        x_i = w_i + \theta_1 w_{i-1}

    where w_i are white noise ~ N(0,1).

    Parameters
    ----------
    t1 : float, array_like
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    """
    # Make inputs 2d arrays for broadcasting with w
    t1 = np.asanyarray(t1).reshape((-1, 1))

    random_state = random_state or np.random

    # i.i.d. sequence ~ N(0,1)
    w = random_state.randn(batch_size, n_obs + 2)
    x = w[:, 2:] + t1 * w[:, 1:-1]
    return x.reshape((batch_size, -1))  # ensure 2D


def stochastic_volatility(w=-0.736,
                          rho=0.9,
                          sigma_v=0.36,
                          n_obs=100,
                          batch_size=1,
                          random_state=None):
    """Sample for a stochastic volatility model.

    specified in Frazier and Drovandi (2021). This is the true Data
    Generating Process for this example.
    Uses a normally distributed shock term.

    Parameters
    ----------
    w : float, optional
    rho : float, optional
    sigma_v : float, optional
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
    y_mat : np.array

    """
    random_state = random_state or np.random

    h_mat = np.zeros((batch_size, n_obs))
    y_mat = np.zeros((batch_size, n_obs))

    w_vec = np.repeat(w, batch_size)
    rho_vec = np.repeat(rho, batch_size)
    sigma_v_vec = np.repeat(sigma_v, batch_size)

    h_mat[:, 0] = w_vec + random_state.normal(0, 1, batch_size) * sigma_v_vec
    y_mat[:, 0] = np.exp(h_mat[:, 0]/2) * random_state.normal(0, 1, batch_size)

    for i in range(n_obs - 1):
        h_mat[:, i] = w_vec + rho_vec * h_mat[:, i-1] + \
            random_state.normal(0, 1, batch_size) * sigma_v_vec
        y_mat[:, i] = np.exp(h_mat[:, i]/2)*random_state.normal(0, 1, batch_size)

    return y_mat.reshape((batch_size, -1))  # ensure 2d


def autocov(x, lag=0):
    """Return the autocovariance.

    Assumes a (weak) univariate stationary process with mean 0.
    Realizations are in rows.

    Parameters
    ----------
    x : np.array of size (n, m)
    lag : int, optional

    Returns
    -------
    C : np.array of size (n,)

    """
    x = np.atleast_2d(x)
    # In R this is normalized with x.shape[1]
    if lag == 0:
        C = np.mean(x[:, :] ** 2, axis=1)
    else:
        C = np.mean(x[:, lag:] * x[:, :-lag], axis=1)

    return C


def get_model(n_obs=50, true_params=None, seed_obs=None):
    """Return a complete misspecified MA1 model in inference task.

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
        true_params = [-0.736, 0.9, 0.36]

    y = stochastic_volatility(*true_params, n_obs=n_obs,
                              random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(MA1, n_obs=n_obs)

    m = elfi.ElfiModel()
    elfi.Prior('uniform', -1, 2, model=m, name='t1')
    elfi.Simulator(sim_fn, m['t1'], observed=y, name='MA1')
    elfi.Summary(autocov, m['MA1'], name='S1')
    elfi.Summary(autocov, m['MA1'], 1, name='S2')
    elfi.SyntheticLikelihood("bsl", m['S1'], m['S2'], name="SL")
    return m
