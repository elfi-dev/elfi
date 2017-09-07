"""An example implementation of the univariate g-and-k model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi
from elfi.examples.gnk import euclidean_multidim, ss_octile, ss_order, ss_robust

EPS = np.finfo(float).eps


def BiGNK(a1, a2, b1, b2, g1, g2, k1, k2, rho, c=.8, n_obs=150, batch_size=1,
          random_state=None):
    """Sample the bi g-and-k distribution.

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
    a1 : float or array_like
        The location (the 1st dimension).
    a2 : float or array_like
        The location (the 2nd dimension).
    b1 : float or array_like
        The scale (the 1st dimension).
    b2 : float or array_like
        The scale (the 2nd dimension).
    g1 : float or array_like
        The skewness (the 1st dimension).
    g2 : float or array_like
        The skewness (the 2nd dimension).
    k1 : float or array_like
        The kurtosis (the 1st dimension).
    k2 : float or array_like
        The kurtosis (the 2nd dimension).
    rho : float or array_like
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
    a = np.hstack((a1, a2))
    b = np.hstack((b1, b2))
    g = np.hstack((g1, g2))
    k = np.hstack((k1, k2))

    # Sampling from the z term, Equation 3 [1].
    z = []
    for i in range(batch_size):
        matrix_cov = np.array([[1, rho[i]], [rho[i], 1]])
        z_el = ss.multivariate_normal.rvs(cov=matrix_cov,
                                          size=(n_obs),
                                          random_state=random_state)
        z.append(z_el)
    z = np.array(z)

    # Obtaining the first bracket term, Equation 3 [1].
    gdotz = np.einsum('ik,ijk->ijk', g, z)
    term_exp = (1 - np.exp(-gdotz)) / (1 + np.exp(-gdotz))
    term_first = np.einsum('ik,ijk->ijk', b, (1 + c * (term_exp)))

    # Obtaining the second bracket term, Equation 3 [1].
    term_second_unraised = 1 + np.power(z, 2)
    k = np.repeat(k, n_obs, axis=2)
    k = np.swapaxes(k, 1, 2)
    term_second = np.power(term_second_unraised, k)

    # Yielding Equation 3, [1].
    term_product = term_first * term_second * z
    term_product_misaligned = np.swapaxes(term_product, 1, 0)
    y_misaligned = np.add(a, term_product_misaligned)
    y = np.swapaxes(y_misaligned, 1, 0)
    # print(y.shape)
    return y


def get_model(n_obs=150, true_params=None, stats_summary=None, seed_obs=None):
    """Return an initialised bivariate g-and-k model.

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

    # Initialising the default parameter settings as given in [1].
    if true_params is None:
        true_params = [3, 4, 1, 0.5, 1, 2, .5, .4, 0.6]
    if stats_summary is None:
        stats_summary = ['ss_robust']

    # Initialising the default prior settings as given in [1].
    elfi.Prior('uniform', 0, 5, model=m, name='a1')
    elfi.Prior('uniform', 0, 5, model=m, name='a2')
    elfi.Prior('uniform', 0, 5, model=m, name='b1')
    elfi.Prior('uniform', 0, 5, model=m, name='b2')
    elfi.Prior('uniform', -5, 10, model=m, name='g1')
    elfi.Prior('uniform', -5, 10, model=m, name='g2')
    elfi.Prior('uniform', -.5, 5.5, model=m, name='k1')
    elfi.Prior('uniform', -.5, 5.5, model=m, name='k2')
    elfi.Prior('uniform', -1 + EPS, 2 - 2 * EPS, model=m, name='rho')

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
