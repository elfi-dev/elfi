"""Implementation of the alpha-stable stochastic volatility model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def shock_term(alpha, beta, kappa, eta, batch_size=1, random_state=None):
    """Shock term used here is the levy_stable distribution.

    Parameters
    ----------
    alpha : np.array of floats
        Controls the tail heaviness.
    beta : np.array of floats.
        Controls the skewness.
    kappa : np.array of floats
        Controls the scale.
    eta  : np.array of floats
        Controls the location.
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
    v_t : np.array of np.float64

    """
    scipy_randomGen = ss.levy_stable
    scipy_randomGen.random_state = random_state
    v_t = scipy_randomGen.rvs(alpha=alpha,
                              beta=beta,
                              loc=eta,
                              scale=kappa,
                              size=batch_size)
    return v_t


def alpha_stochastic_volatility_model(alpha,
                                      beta,
                                      mu=5,
                                      n_obs=50,
                                      batch_size=1,
                                      random_state=None
                                      ):
    """Sample the alpha-stable SVM distribution.

    Parameters
    ----------
    alpha : np.array of floats
        Controls the tail heaviness.
    beta : np.array of floats
        Controls the skewness.
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
    y_mat : np.array of floats
        Observations of an alpha-stable SVM.

    """
    random_state = random_state or np.random

    # currently assumes remaining parameters are known and fixed
    mu = 5
    phi = 1
    kappa = 1
    eta = 0
    sigma = 0.2

    y_mat = np.zeros((batch_size, n_obs))
    # first time step (does not rely on prev xx_t)
    v_0 = shock_term(alpha, beta, kappa, eta, batch_size, random_state)
    x_0 = random_state.normal(mu+phi*-mu, sigma, batch_size)
    y_mat[:, 0] = x_0*v_0  # assumes x_0 has no prev.
    x_prev = x_0
    for t in range(1, n_obs):
        # draw log volatility term (x_t)
        x_t = random_state.normal(mu+phi*(x_prev-mu), sigma, batch_size)
        # draw shock term from stable distribution
        v_t = shock_term(alpha, beta, kappa, eta, batch_size, random_state)
        y_mat[:, t] = x_t * v_t
        x_prev = x_t
    if batch_size == 1:
        y_mat = y_mat.flatten()
    return y_mat


def identity(x):
    """Return observations as summary."""
    return x


def get_model(n_obs=50, true_params=None, seed_obs=None, parallelise=False,
              n_processes=4):
    """Return a complete alpha-stochastic volatility model in inference task.

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
        true_params = [1.2, 0.5]

    if not parallelise:
        n_processes = 1

    m = elfi.ElfiModel()
    y_obs = alpha_stochastic_volatility_model(*true_params,
                                              n_obs=n_obs,
                                              random_state=np.random.RandomState(seed_obs))

    simulator = partial(alpha_stochastic_volatility_model, n_obs=n_obs)

    elfi.Prior('uniform', 0, 2, model=m, name='alpha')
    elfi.Prior('uniform', 0, 1, model=m, name='beta')
    elfi.Simulator(simulator, m['alpha'], m['beta'],
                   observed=y_obs, name='a_svm', parallelise=parallelise,
                   n_processes=n_processes)
    elfi.Summary(identity,  m['a_svm'], name="identity")
    # NOTE: SVM written for BSL, distance node included but not well tested
    elfi.Distance('euclidean', m['identity'], name='d')
    elfi.SyntheticLikelihood("bsl", m['identity'], name="SL")

    return m
