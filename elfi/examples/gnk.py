"""An example implementation of the bivariate g-and-k model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi

EPS = np.finfo(float).eps


def GNK(a, b, g, k, c=0.8, n_obs=50, batch_size=1, random_state=None):
    """Sample the univariate g-and-k distribution.

    References
    ----------
    [1] Drovandi, Christopher C., and Anthony N. Pettitt. "Likelihood-free
    Bayesian estimation of multivariate quantile distributions."
    Computational Statistics & Data Analysis 55.9 (2011): 2541-2556.
    [2] Allingham, David, R. AR King, and Kerrie L. Mengersen. "Bayesian
    estimation of quantile distributions."Statistics and Computing 19.2
    (2009): 189-201.

    Parameters
    ----------
    a : float or array_like
        The location.
    b : float or array_like
        The scale.
    g : float or array_like
        The skewness.
    k : float or array_like
        The kurtosis.
    c : float, optional
        The overall asymmetry parameter, as a convention fixed to 0.8 [2].
    n_obs : int, optional
        The number of the observed points
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    array_like
        The yielded points.

    """
    # Standardising the parameters
    a = np.asanyarray(a).reshape((-1, 1))
    b = np.asanyarray(b).reshape((-1, 1))
    g = np.asanyarray(g).reshape((-1, 1))
    k = np.asanyarray(k).reshape((-1, 1))

    # Sampling from the z term, Equation 1, [2].
    z = ss.norm.rvs(size=(batch_size, n_obs), random_state=random_state)

    # Yielding Equation 1, [2].
    term_exp = (1 - np.exp(-g * z)) / (1 + np.exp(-g * z))
    y = a + b * (1 + c * (term_exp)) * (1 + z**2)**k * z

    # Dedicating an axis for the data dimensionality.
    y = np.expand_dims(y, axis=2)
    return y


def get_model(n_obs=50, true_params=None, stats_summary=None, seed_obs=None):
    """Return an initialised univariate g-and-k model.

    Parameters
    ----------
    n_obs : int, optional
        The number of the observed points.
    true_params : array_like, optional
        The parameters defining the model.
    stats_summary : array_like, optional
        The chosen summary statistics, expressed as a list of strings.
        Options: ['ss_order'], ['ss_robust'], ['ss_octile'].
    seed_obs : np.random.RandomState, optional

    Returns
    -------
    elfi.ElfiModel

    """
    m = elfi.ElfiModel()

    # Initialising the default parameter settings as given in [2].
    if true_params is None:
        true_params = [3, 1, 2, .5]
    if stats_summary is None:
        stats_summary = ['ss_order']

    # Initialising the default prior settings as given in [2].
    elfi.Prior('uniform', 0, 10, model=m, name='a')
    elfi.Prior('uniform', 0, 10, model=m, name='b')
    elfi.Prior('uniform', 0, 10, model=m, name='g')
    elfi.Prior('uniform', 0, 10, model=m, name='k')

    # Generating the observations.
    y_obs = GNK(*true_params, n_obs=n_obs,
                random_state=np.random.RandomState(seed_obs))

    # Defining the simulator.
    fn_sim = partial(GNK, n_obs=n_obs)
    elfi.Simulator(fn_sim, m['a'], m['b'], m['g'], m['k'], observed=y_obs,
                   name='GNK')

    # Initialising the chosen summary statistics.
    fns_summary_all = [ss_order, ss_robust, ss_octile]
    fns_summary_chosen = []
    for fn_summary in fns_summary_all:
        if fn_summary.__name__ in stats_summary:
            summary = elfi.Summary(fn_summary, m['GNK'],
                                   name=fn_summary.__name__)
            fns_summary_chosen.append(summary)

    elfi.Discrepancy(euclidean_multidim, *fns_summary_chosen, name='d')

    return m


def euclidean_multidim(*simulated, observed):
    """Calculate the multi-dimensional Euclidean distance.

    Parameters
    ----------
    *simulated: array_like
        The simulated points.
    observed : array_like
        The observed points.

    Returns
    -------
    array_like

    """
    pts_sim = np.column_stack(simulated)
    pts_obs = np.column_stack(observed)
    d_multidim = np.sum((pts_sim - pts_obs)**2., axis=1)
    d_squared = np.sum(d_multidim, axis=1)
    d = np.sqrt(d_squared)

    return d


def ss_order(y):
    """Obtain the order summary statistic, [2].

    The statistic reaches an optimal performance upon a low number of
    observations.

    Parameters
    ----------
    y : array_like
        The yielded points.

    Returns
    -------
    array_like

    """
    ss_order = np.sort(y)

    return ss_order


def ss_robust(y):
    """Obtain the robust summary statistic, [1].

    The statistic reaches an optimal performance upon a high number of
    observations.

    Parameters
    ----------
    y : array_like
        The yielded points.

    Returns
    -------
    array_like

    """
    ss_a = _get_ss_a(y)
    ss_b = _get_ss_b(y)
    ss_g = _get_ss_g(y)
    ss_k = _get_ss_k(y)

    ss_robust = np.stack((ss_a, ss_b, ss_g, ss_k), axis=1)

    return ss_robust


def ss_octile(y):
    """Obtain the octile summary statistic.

    The statistic reaches an optimal performance upon a high number of
    observations. As reported in [1], it is more stable than ss_robust.

    Parameters
    ----------
    y : array_like
        The yielded points.

    Returns
    -------
    array_like

    """
    octiles = np.linspace(12.5, 87.5, 7)
    E1, E2, E3, E4, E5, E6, E7 = np.percentile(y, octiles, axis=1)

    ss_octile = np.stack((E1, E2, E3, E4, E5, E6, E7), axis=1)

    return ss_octile


def _get_ss_a(y):
    L2 = np.percentile(y, 50, axis=1)
    ss_a = L2

    return ss_a


def _get_ss_b(y):
    L1, L3 = np.percentile(y, [25, 75], axis=1)
    ss_b = L3 - L1

    # Adjusting the zero values to avoid division issues.
    ss_b_ravelled = ss_b.ravel()
    idxs_zero = np.where(ss_b_ravelled == 0)[0]
    ss_b_ravelled[idxs_zero] += EPS
    n_dim = y.shape[-1]
    n_batches = y.shape[0]
    ss_b = ss_b_ravelled.reshape(n_batches, n_dim)

    return ss_b


def _get_ss_g(y):
    L1, L2, L3 = np.percentile(y, [25, 50, 75], axis=1)

    ss_b = _get_ss_b(y)
    ss_g = np.divide(L3 + L1 - 2 * L2, ss_b)

    return ss_g


def _get_ss_k(y):
    E1, E3, E5, E7 = np.percentile(y, [12.5, 37.5, 62.5, 87.5], axis=1)

    ss_b = _get_ss_b(y)
    ss_k = np.divide(E7 - E5 + E3 - E1, ss_b)

    return ss_k
