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
        Controls the tail heaviness, 0 < alpha <= 2.
    beta : np.array of floats.
        Controls the skewness, -1 <= beta <= 1.
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
    distribution = ss.levy_stable(alpha=alpha, beta=beta, loc=eta, scale=kappa)
    distribution.dist.parameterization = 'S0'
    distribution.random_state = random_state
    v_t = distribution.rvs(size=(n_obs, batch_size))
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
                                      kappa,
                                      eta,
                                      mu,
                                      phi,
                                      sigma,
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
    kappa : float or np.array
        Controls the shock term distribution scale.
    eta  : float or np.array
        Controls the shock term distribution location.
    mu : float or np.array, optional
        Log-volatility model mean.
    phi : float or np.array
        Log-volatility model persistence parameter, -1 < phi < 1.
    sigma : float or np.array
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


def kurt(x):
    """Calculate quantile-based kurtosis measure.

    Parameters
    ----------
    x : np.array in shape (batch_size, n_obs)

    Returns
    -------
    np.array in shape (batch_size, 1)

    """
    qs = np.quantile(x, q=[0.05, 0.25, 0.75, 0.95], axis=1)
    return np.transpose((qs[3] - qs[0])/(qs[2] - qs[1]))


def skew(x):
    """Calculate quantile-based skewness measure.

    Parameters
    ----------
    x : np.array in shape (batch_size, n_obs)

    Returns
    -------
    np.array in shape (batch_size, 1)

    """
    qs = np.quantile(x, q=[0.05, 0.50, 0.95], axis=1)
    return np.transpose((((qs[2] - qs[1]) - (qs[1] - qs[0]))/(qs[2] - qs[0])))


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
    # Unknown parameters include the stable distribution parameters alpha and beta
    if true_params is None:
        true_params = [1.2, 0.5]

    # Remaining parameters are assumed known and fixed
    fixed = {'kappa': 1, 'eta': 0, 'mu': 0, 'phi': 0.95, 'sigma': 0.2}

    y_obs = alpha_stochastic_volatility_model(*true_params, **fixed,
                                              n_obs=n_obs,
                                              random_state=np.random.RandomState(seed_obs))

    simulator = partial(alpha_stochastic_volatility_model, n_obs=n_obs)

    m = elfi.ElfiModel()
    elfi.Prior('uniform', 0.5, 1.5, model=m, name='alpha')
    elfi.Prior('uniform', -1, 2, model=m, name='beta')
    constants = [elfi.Constant(value, model=m, name=param) for param, value in fixed.items()]
    elfi.Simulator(simulator, m['alpha'], m['beta'], *constants, observed=y_obs, name='a_svm')
    # NOTE: SVM written for BSL, distance node included but not well tested
    elfi.Summary(kurt, m['a_svm'], name='kurt')
    elfi.Summary(skew, m['a_svm'], name='skew')
    elfi.Distance('euclidean', m['kurt'], m['skew'], name='d')

    logger.info("Generated observations with true parameters "
                "alpha: %.1f, beta: %.1f", *true_params)

    return m
