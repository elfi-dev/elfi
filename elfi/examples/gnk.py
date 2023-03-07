"""Implementation of the univariate g-and-k example model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def GNK(A, B, g, k, c=0.8, n_obs=50, batch_size=1, random_state=None):
    """Sample the univariate g-and-k distribution.

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
    A : float or array_like
        Location parameter.
    B : float or array_like
        Scale parameter.
    g : float or array_like
        Skewness parameter.
    k : float or array_like
        Kurtosis parameter.
    c : float, optional
        Overall asymmetry parameter, by default fixed to 0.8 as in Allingham et al. (2009).
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    array_like
        Yielded points (the array's shape corresponds to (batch_size, n_points, n_dims).

    """
    # Transforming the arrays' shape to be compatible with batching.
    A = np.asanyarray(A).reshape((-1, 1))
    B = np.asanyarray(B).reshape((-1, 1))
    g = np.asanyarray(g).reshape((-1, 1))
    k = np.asanyarray(k).reshape((-1, 1))

    # Obtaining z(p) ~ N(0, 1).
    z = ss.norm.rvs(size=(batch_size, n_obs), random_state=random_state)

    # Evaluating the quantile function Q_{gnk}.
    y = A + B * (1 + c * ((1 - np.exp(-g * z)) / (1 + np.exp(-g * z)))) * (1 + z**2)**k * z

    # Dedicating a dummy axis for the dimensionality of the points.
    y = y[:, :, np.newaxis]
    return y


def get_model(n_obs=50, true_params=None, seed=None):
    """Initialise the g-and-k model.

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

    # Initialising the parameters as in Allingham et al. (2009).
    if true_params is None:
        true_params = [3, 1, 2, .5]

    # Initialising the prior settings as in Allingham et al. (2009).
    priors = []
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='A'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='B'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='g'))
    priors.append(elfi.Prior('uniform', 0, 10, model=m, name='k'))

    # Obtaining the observations.
    y_obs = GNK(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed))

    # Defining the simulator.
    fn_simulator = partial(GNK, n_obs=n_obs)
    elfi.Simulator(fn_simulator, *priors, observed=y_obs, name='GNK')

    # Initialising the summary statistics as in Allingham et al. (2009).
    default_ss = elfi.Summary(ss_order, m['GNK'], name='ss_order')
    # Using the multi-dimensional Euclidean distance function as
    # the summary statistics' implementations are designed for multi-dimensional cases.
    elfi.Discrepancy(euclidean_multiss, default_ss, name='d')
    return m


def euclidean_multiss(*simulated, observed):
    """Calculate the Euclidean distances merging summary statistics.

    The shape of the arrays corresponds to (batch_size, dim_ss, dim_ss_point), where
    dim_ss corresponds to the dimensionality of the summary statistics, and
    dim_ss_point corresponds to the dimensionality a summary statistic data point.

    Parameters
    ----------
    *simulated: array_like
    observed : array_like

    Returns
    -------
    array_like

    """
    pts_sim = simulated[0]
    pts_obs = observed[0]

    # Integrating over the summary statistics.
    d_ss_merged = np.sum((pts_sim - pts_obs)**2., axis=1)

    # Integrating over the summary statistics' data point dimensionality.
    d_ss_point_merged = np.sum(d_ss_merged, axis=1)

    d = np.sqrt(d_ss_point_merged)
    return d


def ss_order(y):
    """Obtain the order summary statistic described in Allingham et al. (2009).

    The statistic reaches the optimal performance upon a low number of observations.

    Parameters
    ----------
    y : array_like
        Yielded points.

    Returns
    -------
    array_like of the shape (batch_size, dim_ss=len(y), dim_ss_point)

    """
    ss_order = np.sort(y)
    return ss_order


def ss_robust(y):
    """Obtain the robust summary statistic described in Drovandi and Pettitt (2011).

    The statistic reaches the optimal performance upon a high number of
    observations.

    Parameters
    ----------
    y : array_like
        Yielded points.

    Returns
    -------
    array_like of the shape (batch_size, dim_ss=4, dim_ss_point)

    """
    ss_A = _get_ss_A(y)
    ss_B = _get_ss_B(y)
    ss_g = _get_ss_g(y)
    ss_k = _get_ss_k(y)

    # Combining the summary statistics.
    ss_robust = np.hstack((ss_A, ss_B, ss_g, ss_k))
    ss_robust = ss_robust[:, :, np.newaxis]
    return ss_robust


def ss_octile(y):
    """Obtain the octile summary statistic.

    The statistic reaches the optimal performance upon a high number of
    observations. According to Allingham et al. (2009), it is more stable than ss_robust.

    Parameters
    ----------
    y : array_like
        Yielded points.

    Returns
    -------
    array_like of the shape (batch_size, dim_ss=8, dim_ss_point)

    """
    octiles = np.linspace(12.5, 87.5, 7)
    E1, E2, E3, E4, E5, E6, E7 = np.percentile(y, octiles, axis=1)

    # Combining the summary statistics.
    ss_octile = np.hstack((E1, E2, E3, E4, E5, E6, E7))
    ss_octile = ss_octile[:, :, np.newaxis]
    return ss_octile


def _get_ss_A(y):
    L2 = np.percentile(y, 50, axis=1)
    ss_A = L2
    return ss_A


def _get_ss_B(y):
    L1, L3 = np.percentile(y, [25, 75], axis=1)

    # Avoiding the zero value (ss_B is used for division).
    ss_B = (L3 - L1).ravel()
    idxs_zero = np.where(ss_B == 0)[0]
    ss_B[idxs_zero] += np.finfo(float).eps

    # Transforming the summary statistics back into the compatible shape.
    n_dim = y.shape[-1]
    n_batches = y.shape[0]
    ss_B = ss_B.reshape(n_batches, n_dim)
    return ss_B


def _get_ss_g(y):
    L1, L2, L3 = np.percentile(y, [25, 50, 75], axis=1)
    ss_B = _get_ss_B(y)
    ss_g = np.divide(L3 + L1 - 2 * L2, ss_B)
    return ss_g


def _get_ss_k(y):
    E1, E3, E5, E7 = np.percentile(y, [12.5, 37.5, 62.5, 87.5], axis=1)
    ss_B = _get_ss_B(y)
    ss_k = np.divide(E7 - E5 + E3 - E1, ss_B)
    return ss_k
