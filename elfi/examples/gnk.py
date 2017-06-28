import numpy as np
import scipy.stats as ss
import elfi
from functools import partial


"""An example implementation of the univariate g-and-k model.
"""

EPS = np.finfo(float).eps


def GNK(A, B, g, k, c=0.8, n_obs=50, batch_size=25, random_state=None):
    """The univariate g-and-k model.

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
    A : float
        The location.
    B : float
        The scale.
    g : float
        The skewness.
    k : float
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
    A = np.asanyarray(A).reshape((-1, 1))
    B = np.asanyarray(B).reshape((-1, 1))
    g = np.asanyarray(g).reshape((-1, 1))
    k = np.asanyarray(k).reshape((-1, 1))

    # Sampling from the z term, Equation 1, [2].
    z = ss.norm.rvs(size=(batch_size, n_obs), random_state=random_state)

    # Yielding Equation 1, [2].
    term_exp = (1 - np.exp(-g * z)) / (1 + np.exp(-g * z))
    y = A + B * (1 + c * (term_exp)) * (1 + z**2)**k * z

    return y


def get_model(n_obs=50, true_params=None, stats_summary=None, seed_obs=None):
    """Returns an initialised univariate g-and-k model.

    Parameters
    ----------
    n_obs : int, optional
        The number of the observed points.
    true_params : array_like, optional
        The parameters defining the model.
    stats_summary : array_like, optional
        The chosen summary statistics, expressed as a list of strings.
    seed_obs : np.random.RandomState, optional

    Returns
    -------
    elfi.ElfiModel
    """
    m = elfi.ElfiModel(set_current=False)

    # Initialising the default parameter settings as given in [2].
    if true_params is None:
        true_params = [3, 1, 2, .5]
    if stats_summary is None:
        # stats_summary = ['ss_order']
        stats_summary = ['ss_robust']
        # stats_summary = ['ss_octile']

    # Initialising the default prior settings as given in [2].
    elfi.Prior('uniform', 0, 10, model=m, name='A')
    elfi.Prior('uniform', 0, 10, model=m, name='B')
    elfi.Prior('uniform', 0, 10, model=m, name='g')
    elfi.Prior('uniform', 0, 10, model=m, name='k')

    y_obs = GNK(*true_params, n_obs=n_obs,
                random_state=np.random.RandomState(seed_obs))

    fn_sim = partial(GNK, n_obs=n_obs)
    elfi.Simulator(fn_sim, m['A'], m['B'], m['g'], m['k'], observed=y_obs,
                   name='GNK')

    # Initialising the chosen summary statistics.
    fns_summary_all = [ss_order, ss_robust, ss_octile]
    fns_summary_chosen = []
    for fn_summary in fns_summary_all:
        if fn_summary.__name__ in stats_summary:
            summary = elfi.Summary(fn_summary, m['GNK'],
                                   name=fn_summary.__name__)
            fns_summary_chosen.append(summary)

    elfi.Distance('euclidean', *fns_summary_chosen, name='d')

    return m


def ss_order(y):
    """Obtaining the order summary statistic, [2].
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
    """Obtaining the robust summary statistic, [1].
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
    ss_A = _get_ss_A(y)
    ss_B = _get_ss_B(y)
    ss_g = _get_ss_g(y)
    ss_k = _get_ss_k(y)

    ss_robust = np.stack((ss_A, ss_B, ss_g, ss_k), axis=1)

    return ss_robust


def ss_octile(y):
    """Obtaining the octile summary statistic.
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
    E1 = ss.scoreatpercentile(y, 12.5, axis=1)
    E2 = ss.scoreatpercentile(y, 25, axis=1)
    E3 = ss.scoreatpercentile(y, 37.5, axis=1)
    E4 = ss.scoreatpercentile(y, 50, axis=1)
    E5 = ss.scoreatpercentile(y, 62.5, axis=1)
    E6 = ss.scoreatpercentile(y, 75, axis=1)
    E7 = ss.scoreatpercentile(y, 87.5, axis=1)

    ss_octile = np.stack((E1, E2, E3, E4, E5, E6, E7), axis=1)

    return ss_octile


def _get_ss_A(y):
    L2 = ss.scoreatpercentile(y, 50, axis=1)
    ss_A = L2

    return ss_A


def _get_ss_B(y):
    L1 = ss.scoreatpercentile(y, 25, axis=1)
    L3 = ss.scoreatpercentile(y, 75, axis=1)
    ss_B = L3 - L1
    for idx_el, el in enumerate(ss_B):
            ss_B[idx_el] += EPS

    return ss_B


def _get_ss_g(y):
    L1 = ss.scoreatpercentile(y, 25, axis=1)
    L2 = ss.scoreatpercentile(y, 50, axis=1)
    L3 = ss.scoreatpercentile(y, 75, axis=1)
    ss_B = _get_ss_B(y)
    ss_g = np.divide(L3 + L1 - 2*L2, ss_B)

    return ss_g


def _get_ss_k(y):
    E1 = ss.scoreatpercentile(y, 12.5, axis=1)
    E3 = ss.scoreatpercentile(y, 37.5, axis=1)
    E5 = ss.scoreatpercentile(y, 62.5, axis=1)
    E7 = ss.scoreatpercentile(y, 87.5, axis=1)
    ss_B = _get_ss_B(y)
    ss_k = np.divide(E7 - E5 + E3 - E1, ss_B)

    return ss_k
