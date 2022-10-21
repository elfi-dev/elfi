"""Example implementation of an alpha-stable stochastic volatility model.

References
----------
Priddle and Drovandi (2020) Transformations in Semi-Parametric Bayesian Synthetic Likelihood.
https://arxiv.org/abs/2007.01485

Vankov et al (2019) Filtering and Estimation for a Class of Stochastic Volatility Models with
Intractable Likelihoods. Bayesian Analysis 14(1): 29-52. https://doi.org/10.1214/18-BA1099

"""

import logging
from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def shock_term(alpha, beta, kappa, eta, n_obs, batch_size=1, random_state=None):
    """Sample shock term.

    Shock term used here is the levy_stable distribution.

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
    n_obs : int
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
                              size=(n_obs, batch_size))
    return v_t


def log_vol(mu, phi, sigma, n_obs, prev_x=None, batch_size=1, random_state=None):
    """Sample log-volatilities.

    Log-volatilities are modelled as an AR(1) process expressed in the mean/difference form.

    Parameters
    ----------
    mu : float or np.array
        Mean parameter.
    phi : float or np.array
        Persistence parameter, -1 < phi < 1.
    sigma : float or np.array
        Noise distribution scale.
    n_obs : int
        Number of observations.
    prev_x : float, optional
        Previous observed value, used to initialise the process.
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
        np.array in shape (n_obs, batch_size)

    """
    x = np.zeros((n_obs, batch_size))
    if prev_x is None:
        scale = sigma / np.sqrt((1-np.minimum(np.squeeze(phi)**2, 0.99999)))
        x[0] = ss.norm.rvs(mu, scale, batch_size, random_state=random_state)
    else:
        x[0] = ss.norm.rvs(mu + phi * (prev_x - mu), sigma, batch_size, random_state=random_state)
    for t in range(1, n_obs):
        x[t] = ss.norm.rvs(mu + phi * (x[t-1] - mu), sigma, batch_size, random_state=random_state)
    return x


def alpha_stochastic_volatility_model(alpha,
                                      beta,
                                      kappa=1,
                                      eta=0,
                                      mu=0,
                                      phi=0.95,
                                      sigma=0.2,
                                      n_obs=50,
                                      x_0=None,
                                      batch_size=1,
                                      random_state=None
                                      ):
    """Sample the alpha-stable SVM distribution.

    Parameters
    ----------
    alpha : np.array of floats
        Controls the shock term distribution tail heaviness.
    beta : np.array of floats
        Controls the shock term distribution skewness.
    kappa : float or np.array, optional
        Controls the shock term distribution scale.
    eta  : float or np.array, optional
        Controls the shock term distribution location.
    mu : float or np.array, optional
        Log-volatility model mean.
    phi : float or np.array, optional
        Log-volatility model persistence parameter, -1 < phi < 1.
    sigma : float or np.array, optional
        Log-volatility model noise distribution scale.
    n_obs : int, optional
        Number of observations.
    x_0 : float, optional
        Initial log-volatility.
    batch_size : int, optional
    random_state : RandomState, optional

    Returns
    -------
    y_mat : np.array of floats
        Observations of an alpha-stable SVM.

    """
    # draw log volatility term
    x_t = log_vol(mu, phi, sigma, n_obs, x_0, batch_size, random_state)
    # draw shock term from stable distribution
    v_t = shock_term(alpha, beta, kappa, eta, n_obs, batch_size, random_state)
    # calculate returns
    y_mat = np.exp(0.5 * x_t) * v_t
    return np.transpose(y_mat)


def get_model(n_obs=50, true_params=None, seed_obs=None):
    """Return a complete alpha-stochastic volatility model in inference task.

    Parameters
    ----------
    n_obs : int, optional
        observation length
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
        true_params = [1.2, 0.5]

    m = elfi.ElfiModel()
    y_obs = alpha_stochastic_volatility_model(*true_params,
                                              n_obs=n_obs,
                                              random_state=np.random.RandomState(seed_obs))

    simulator = partial(alpha_stochastic_volatility_model, n_obs=n_obs)

    elfi.Prior('uniform', 0, 2, model=m, name='alpha')
    elfi.Prior('uniform', -1, 2, model=m, name='beta')
    elfi.Simulator(simulator, m['alpha'], m['beta'],
                   observed=y_obs, name='a_svm')
    # NOTE: SVM written for BSL, distance node included but not well tested
    elfi.Distance('euclidean', m['a_svm'], name='d')

    logger.info("Generated observations with true parameters "
                "alpha: %.1f, beta: %.1f", *true_params)

    return m
