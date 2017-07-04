"""The general, n-dimensional Gaussian noise model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def Gauss_nd_mean(*params, var_ij=.5, var_ii=1, n_obs=15, batch_size=1, random_state=None):
    """Sample the Gaussian distribution.

    Notes
    -----
    - The n-dimensional mean corresponds to the model's parameters;
    - The variance is fixed (not a parameter).

    Reference
    ---------
    The default settings replicate the experiment settings used by:
        [1] arXiv:1704.00520 (Järvenpää et al., 2017).

    Parameters
    ----------
    *params : array_like
        The mean of a respective dimension (e.g., the first argument is for
        the first dimension).
    var_ij : float, optional
        The non-diagonal-variance.
    var_ii : int, optional
        The diagonal variance.
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    y : array_like

    """
    n_dim = len(params)
    # Formatting the mean vector (including the batches as rows).
    mu = np.zeros(shape=(batch_size, n_dim))
    for idx_dim, param in enumerate(params):
        mu[:, idx_dim] = param
    # Formatting the covariance matrix.
    sigma = np.zeros(shape=(n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            if i != j:
                sigma[i, j] = var_ij
            else:
                sigma[i, j] = var_ii
    # Sampling observations.
    y = np.zeros(shape=(batch_size, n_obs, n_dim))
    for idx_batch, mu_batch in enumerate(mu):
        y_batch = ss.multivariate_normal.rvs(mean=mu_batch,
                                             cov=sigma,
                                             size=(n_obs),
                                             random_state=random_state)
        if n_dim == 1:
            y_batch = y_batch[:, None]
        y[idx_batch, :, :] = y_batch

    # Storing a global covariance variable for the Mahalanobis distance.
    global cov
    cov = np.linalg.inv(sigma)

    return y


def ss_mean(x):
    """Return the summary statistic corresponding to the mean."""
    ss = np.mean(x, axis=1)
    return ss


def ss_var(x):
    """Return the summary statistic corresponding to the variance."""
    ss = np.var(x, axis=1)
    return ss


def get_model(var_ij=.5, var_ii=1, n_obs=15, true_params=None, seed_obs=None):
    """Return an initialised Gaussian noise model.

    Parameters
    ----------
    n_obs : int
    true_params : array_like
        The mean of a respective dimension (e.g., the first argument is for
        the first dimension).
    seed_obs : np.random.RandomState, optional

    Returns
    -------
    m : elfi.ElfiModel

    """
    # Using the default settings (the 2-D Gaussian model).
    if true_params is None:
        true_params = [4, 4]

    y_obs = Gauss_nd_mean(*true_params,
                          var_ij=var_ij,
                          var_ii=var_ii,
                          n_obs=n_obs,
                          random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(Gauss_nd_mean, n_obs=n_obs)

    # Defining the parameters' priors (n dimensions).
    m = elfi.ElfiModel()
    priors = []
    for i in range(len(true_params)):
        name_prior = 'mu_{}'.format(i)
        prior = elfi.Prior('uniform', 0, 8, model=m, name=name_prior)
        priors.append(prior)

    elfi.Simulator(sim_fn, *priors, observed=y_obs, name='Gauss_nd_mean')
    elfi.Summary(ss_mean, m['Gauss_nd_mean'], name='ss_mean')
    elfi.Summary(ss_var, m['Gauss_nd_mean'], name='ss_var')
    elfi.Discrepancy(mahalanobis, m['ss_mean'], m['ss_var'], name='d')
    return m


def mahalanobis(*simulated, observed):
    """Calculate the Mahalanobis distance."""
    means_sim = simulated[0]
    vars_sim = simulated[1]

    means_obs = observed[0]
    vars_obs = observed[1]

    d = []
    for idx in range(len(means_sim)):
        d_mean = np.sqrt((means_obs - means_sim[idx]).dot(cov).dot((means_obs - means_sim[idx]).T))
        d_var = np.sqrt((vars_obs - vars_sim[idx]).dot(cov).dot((vars_obs - vars_sim[idx]).T))
        d.append(d_mean + d_var)
    d = np.array(d)
    return d
