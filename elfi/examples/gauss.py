import numpy as np
import scipy.stats as ss
import elfi
from functools import partial


"""An example implementation of a Gaussian noise model.
"""


def Gauss(mu, sigma, n_obs=50, batch_size=1, random_state=None):
    # Standardising the parameter's format.
    mu = np.asanyarray(mu).reshape((-1, 1))
    sigma = np.asanyarray(sigma).reshape((-1, 1))
    y = ss.norm.rvs(loc=mu, scale=sigma, size=(batch_size, n_obs),
        random_state=random_state)
    return y


def ss_mean(x):
    """The summary statistic corresponding to the mean.
    """
    ss = np.mean(x, axis=1)
    return ss


def ss_var(x):
    """The summary statistic corresponding to the variance.
    """
    ss = np.var(x, axis=1)
    return ss


def get_model(n_obs=50, true_params=None, seed_obs=None):
    """Returns a complete Gaussian noise model

    Parameters
    ----------
    n_obs : int
        the number of observations
    true_params : list
        true_params[0] corresponds to the mean,
        true_params[1] corresponds to the standard deviation
    seed_obs : None, int
        seed for the observed data generation

    Returns
    -------
    m : elfi.ElfiModel
    """

    if true_params is None:
        true_params = [10, 2]

    y_obs = Gauss(*true_params, n_obs=n_obs,
        random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(Gauss, n_obs=n_obs)

    m = elfi.ElfiModel()
    elfi.Prior('uniform', -10, 50, model=m, name='mu')
    elfi.Prior('truncnorm', 0.01, 5, model=m, name='sigma')
    elfi.Simulator(sim_fn, m['mu'], m['sigma'], observed=y_obs, name='Gauss')
    elfi.Summary(ss_mean, m['Gauss'], name='S1')
    elfi.Summary(ss_var, m['Gauss'], name='S2')
    elfi.Distance('euclidean', m['S1'], m['S2'], name='d')

    return m
