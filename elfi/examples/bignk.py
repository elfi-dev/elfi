"""Implementation of the bivariate g-and-k example model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi
from elfi.examples.gnk import euclidean_multiss, ss_robust


def BiGNK(A1, A2, B1, B2, g1, g2, k1, k2, rho, c=.8, n_obs=150, batch_size=1, random_state=None):
    """Sample the bivariate g-and-k distribution.

    References
    ----------
    [1] Drovandi, C. C., & Pettitt, A. N. (2011).
    Likelihood-free Bayesian estimation of multivariate quantile distributions.
    Computational Statistics & Data Analysis, 55(9), 2541-2556.
    [2] Allingham, D., King, R. A. R., & Mengersen, K. L. (2009).
    Bayesian estimation of quantile distributions.
    Statistics and Computing, 19(2), 189-201.

    The quantile function of g-and-k distribution is defined as follows:

    Q_{gnk} = A + B * (1 + c * (1 - exp(-g * z(p)) / 1 + exp(-g * z(p))))
            * (1 + z(p)^2)^k * z(p), where

    z(p) is the p-th standard normal quantile.

    To sample from the g-and-k distribution, draw z(p) ~ N(0, 1) and evaluate Q_{gnk}.

    Parameters
    ----------
    A1 : float or array_like
        Location parameter (the 1st dimension).
    A2 : float or array_like
        Location parameter (the 2nd dimension).
    B1 : float or array_like
        Scale parameter (the 1st dimension).
    B2 : float or array_like
        Scale parameter (the 2nd dimension).
    g1 : float or array_like
        Skewness parameter (the 1st dimension).
    g2 : float or array_like
        Skewness parameter (the 2nd dimension).
    k1 : float or array_like
        Kurtosis parameter (the 1st dimension).
    k2 : float or array_like
        Kurtosis parameter (the 2nd dimension).
    rho : float or array_like
        Parameters' covariance.
    c : float, optional
        Overall asymmetry parameter, by default fixed to 0.8 as in Allingham et al. (2009).
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    array_like
        Yielded points.

    """
    # Transforming the arrays' shape to be compatible with batching.
    A1 = np.asanyarray(A1).reshape((-1, 1))
    A2 = np.asanyarray(A2).reshape((-1, 1))
    B1 = np.asanyarray(B1).reshape((-1, 1))
    B2 = np.asanyarray(B2).reshape((-1, 1))
    g1 = np.asanyarray(g1).reshape((-1, 1))
    g2 = np.asanyarray(g2).reshape((-1, 1))
    k1 = np.asanyarray(k1).reshape((-1, 1, 1))
    k2 = np.asanyarray(k2).reshape((-1, 1, 1))
    rho = np.asanyarray(rho).reshape((-1, 1))

    # Merging the multi-dimensional parameters.
    A = np.hstack((A1, A2))
    B = np.hstack((B1, B2))
    g = np.hstack((g1, g2))
    k = np.hstack((k1, k2))

    # Obtaining z(p) ~ N(0, 1).
    z_batches = []
    for i in range(batch_size):
        # Initialising a separate covariance matrix for each batch.
        matrix_cov = np.array([[1, float(rho[i])], [float(rho[i]), 1]])

        z_batch = ss.multivariate_normal.rvs(cov=matrix_cov, size=n_obs, random_state=random_state)
        z_batches.append(z_batch)
    z = np.array(z_batches)

    # Obtaining the first bracket term of the quantile function Q_{gnk}.
    gdotz = np.einsum('ik,ijk->ijk', g, z)
    term_exp = (1 - np.exp(-gdotz)) / (1 + np.exp(-gdotz))
    term_first = np.einsum('ik,ijk->ijk', B, (1 + c * (term_exp)))

    # Obtaining the second bracket term, of the quantile function Q_{gnk}.
    term_second_unraised = 1 + np.power(z, 2)
    k = np.repeat(k, n_obs, axis=2)
    k = np.swapaxes(k, 1, 2)
    term_second = np.power(term_second_unraised, k)

    # Evaluating the quantile function Q_{gnk}.
    term_product = term_first * term_second * z
    term_product_misaligned = np.swapaxes(term_product, 1, 0)
    y_misaligned = np.add(A, term_product_misaligned)
    y_obs = np.swapaxes(y_misaligned, 1, 0)
    return y_obs


def get_model(n_obs=150, true_params=None, seed=None):
    """Return an initialised bivariate g-and-k model.

    Parameters
    ----------
    n_obs : int, optional
        Number of the observations.
    true_params : array_like, optional
        Parameters defining the model.
    seed : np.random.RandomState, optional

    Returns
    -------
    elfi.ElfiModel

    """
    m = elfi.new_model()

    # Initialising the parameters as in Drovandi & Pettitt (2011).
    if true_params is None:
        true_params = [3, 4, 1, 0.5, 1, 2, .5, .4, 0.6]

    # Initialising the prior settings as in Drovandi & Pettitt (2011).
    priors = []
    priors.append(elfi.Prior('uniform', 0, 5, model=m, name='a1'))
    priors.append(elfi.Prior('uniform', 0, 5, model=m, name='a2'))
    priors.append(elfi.Prior('uniform', 0, 5, model=m, name='b1'))
    priors.append(elfi.Prior('uniform', 0, 5, model=m, name='b2'))
    priors.append(elfi.Prior('uniform', -5, 10, model=m, name='g1'))
    priors.append(elfi.Prior('uniform', -5, 10, model=m, name='g2'))
    priors.append(elfi.Prior('uniform', -.5, 5.5, model=m, name='k1'))
    priors.append(elfi.Prior('uniform', -.5, 5.5, model=m, name='k2'))
    EPS = np.finfo(float).eps
    priors.append(elfi.Prior('uniform', -1 + EPS, 2 - 2 * EPS, model=m, name='rho'))

    # Obtaining the observations.
    y_obs = BiGNK(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed))

    # Defining the simulator.
    fn_simulator = partial(BiGNK, n_obs=n_obs)
    elfi.Simulator(fn_simulator, *priors, observed=y_obs, name='BiGNK')

    # Initialising the default summary statistics.
    default_ss = elfi.Summary(ss_robust, m['BiGNK'], name='ss_robust')

    # Using the customEuclidean distance function designed for
    # the summary statistics of shape (batch_size, dim_ss, dim_ss_point).
    elfi.Discrepancy(euclidean_multiss, default_ss, name='d')
    return m
