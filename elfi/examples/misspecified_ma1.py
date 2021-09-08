
from functools import partial

import numpy as np
import scipy.stats as ss

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
    # t2 = np.asanyarray(t2).reshape((-1, 1))
    random_state = random_state or np.random

    print('batch_size', batch_size)
    print('n_obs', n_obs)
    # i.i.d. sequence ~ N(0,1)
    w = random_state.randn(batch_size, n_obs + 2)
    x = w[:, 2:] + t1 * w[:, 1:-1] #+ t2 * w[:, :-2]
    return x


def stochastic_volatility(w=-0.736,
                          rho=0.9,
                          sigma_v=0.36,
                          n_obs=100,
                          batch_size=1,
                          random_state=None):
    # TODO: use svm example now?
    random_state = random_state or np.random
    h_0 = np.zeros(batch_size)

    h_mat = np.zeros((batch_size, n_obs))
    y_mat = np.zeros((batch_size, n_obs))

    w_vec = np.repeat(w, batch_size)
    rho_vec = np.repeat(rho, batch_size)
    sigma_v_vec = np.repeat(sigma_v, batch_size)

    h_mat[:, 0] = w_vec + random_state.normal(0, 1, batch_size) * sigma_v_vec
    y_mat[:, 0] = np.exp(h_mat[:, 0]/2) * random_state.normal(0, 1, batch_size)

    # TODO: for loop - bad?
    for i in range(n_obs - 1):
        j = i + 1
        h_mat[:, i] = w_vec + rho_vec * h_mat[:, i-1] + \
            random_state.normal(0, 1, batch_size) * sigma_v_vec
        y_mat[:, i] = np.exp(h_mat[:, i]/2)*random_state.normal(0, 1, batch_size)

    return y_mat


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
    """TODO """
    if true_params is None:
        true_params = [-0.736, 0.9, 0.36]

    y = stochastic_volatility(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(MA1, n_obs=n_obs)

    m = elfi.ElfiModel()
    elfi.Prior('uniform', -1, 2, model=m, name='t1')
    elfi.Simulator(sim_fn, m['t1'], observed=y, name='MA1')
    elfi.Summary(autocov, m['MA1'], name='S0')
    elfi.Summary(autocov, m['MA1'], 1, name='S1')
    return m