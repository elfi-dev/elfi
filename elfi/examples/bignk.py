import numpy as np
import scipy.stats as ss
import elfi
from functools import partial


"""An example implementation of the bivariate g-and-k model.
"""

EPS = np.finfo(float).eps


def BiGNK(a1, a2, b1, b2, g1, g2, k1, k2, rho, c=.8, n_obs=150, batch_size=25,
          random_state=None):
    """The bivariate g-and-k model.

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
    a1 : float
        The location (the 1st dimension).
    a2 : float
        The location (the 2nd dimension).
    b1 : float
        The scale (the 1st dimension).
    b2 : float
        The scale (the 2nd dimension).
    g1 : float
        The skewness (the 1st dimension).
    g2 : float
        The skewness (the 2nd dimension).
    k1 : float
        The kurtosis (the 1st dimension).
    k2 : float
        The kurtosis (the 2nd dimension).
    rho : float
        The dependence between components (dimensions), [1].
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
    a1 = np.asanyarray(a1).reshape((-1, 1))
    a2 = np.asanyarray(a2).reshape((-1, 1))
    b1 = np.asanyarray(b1).reshape((-1, 1))
    b2 = np.asanyarray(b2).reshape((-1, 1))
    g1 = np.asanyarray(g1).reshape((-1, 1))
    g2 = np.asanyarray(g2).reshape((-1, 1))
    k1 = np.asanyarray(k1).reshape((-1, 1, 1))
    k2 = np.asanyarray(k2).reshape((-1, 1, 1))
    rho = np.asanyarray(rho).reshape((-1, 1))
    A = np.hstack((a1, a2))
    B = np.hstack((b1, b2))
    g = np.hstack((g1, g2))
    k = np.hstack((k1, k2))

    # Sampling from the z term, Equation 3 [1].
    z = []
    for rho_i in rho:
        matrix_cov = np.array([[1, rho_i], [rho_i, 1]])
        z_el = ss.multivariate_normal.rvs(cov=matrix_cov, size=(n_obs),
                                          random_state=random_state)
        z.append(z_el)
    z = np.array(z)

    # Obtaining the first bracket term, Equation 3 [1].
    gdotz = np.einsum('ik,ijk->ijk', g, z)
    term_exp = (1 - np.exp(-gdotz)) / (1 + np.exp(-gdotz))
    term_first = np.einsum('ik,ijk->ijk', B, (1 + c * (term_exp)))

    # Obtaining the second bracket term, Equation 3 [1].
    term_second_unraised = 1 + np.power(z, 2)
    k = np.repeat(k, n_obs, axis=2)
    k = np.swapaxes(k, 1, 2)
    term_second = np.power(term_second_unraised, k)

    # Yielding Equation 3, [1].
    term_product = term_first * term_second * z
    term_product_misaligned = np.swapaxes(term_product, 1, 0)
    y_misaligned = np.add(A, term_product_misaligned)
    y = np.swapaxes(y_misaligned, 1, 0)

    return y


def get_model(n_obs=150, true_params=None, stats_summary=None, seed_obs=None):
    """Returns an initialised bivariate g-and-k model.

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

    # Initialising the default parameter settings as given in [1].
    if true_params is None:
        true_params = [3, 4, 1, 0.5, 1, 2, .5, .4, 0.6]
    if stats_summary is None:
        # stats_summary = ['ss_order']
        stats_summary = ['ss_robust']
        # stats_summary = ['ss_octile']

    # Initialising the default prior settings as given in [1].
    elfi.Prior('uniform', 0, 5, model=m, name='a1')
    elfi.Prior('uniform', 0, 5, model=m, name='a2')
    elfi.Prior('uniform', 0, 5, model=m, name='b1')
    elfi.Prior('uniform', 0, 5, model=m, name='b2')
    elfi.Prior('uniform', -5, 10, model=m, name='g1')
    elfi.Prior('uniform', -5, 10, model=m, name='g2')
    elfi.Prior('uniform', -.5, 5.5, model=m, name='k1')
    elfi.Prior('uniform', -.5, 5.5, model=m, name='k2')
    elfi.Prior('uniform', -1 + EPS, 2 - 2*EPS, model=m, name='rho')

    # Generating the observations.
    y_obs = BiGNK(*true_params, n_obs=n_obs,
                  random_state=np.random.RandomState(seed_obs))

    # Defining the simulator.
    fn_sim = partial(BiGNK, n_obs=n_obs)
    elfi.Simulator(fn_sim, m['a1'], m['a2'], m['b1'], m['b2'], m['g1'],
                   m['g2'], m['k1'], m['k2'], m['rho'], observed=y_obs,
                   name='BiGNK')

    # Initialising the chosen summary statistics.
    fns_summary_all = [ss_order, ss_robust, ss_octile]
    fns_summary_chosen = []
    for fn_summary in fns_summary_all:
        if fn_summary.__name__ in stats_summary:
            summary = elfi.Summary(fn_summary, m['BiGNK'],
                                   name=fn_summary.__name__)
            fns_summary_chosen.append(summary)

    # Defining the distance metric.
    elfi.Discrepancy(euclidean_multidim, *fns_summary_chosen, name='d')

    return m


def euclidean_multidim(*simulated, observed):
    """Calculating the multi-dimensional Euclidean distance.

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
        for idx_dim, el_dim in enumerate(el):
            if el_dim == 0:
                ss_B[idx_el][idx_dim] += EPS

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
