import itertools

import numpy as np
import scipy.stats as ss

import elfi
from elfi.methods.utils import cov2corr
from .empirical_density import estimate_densities, EmpiricalDensity


def _raise(err):
    """Exception raising closure."""
    def fun():
        raise err
    return fun


class MetaGaussian(object):
    """A meta-Gaussian distribution

    Parameters
    ----------
    corr : np.ndarray
        a correlation matrix
    cov : np.ndarray
        a covariance matrix
    marginals : density_like
        a list of objects that implement 'cdf' and 'ppf' methods
    marginal_samples : np.ndarray
        a nxm array of samples where n is the number of observations
        and m is the number of dimensions

    Attributes
    ----------
    corr : np.ndarray
        the correlation matrix
    cov : np.ndarray
        the covariance matrix
    marginals : List
        a list of marginal densities

    References
    ----------
    Jingjing Li, David J. Nott, Yanan Fan, Scott A. Sisson (2016)
    Extending approximate Bayesian computation methods to high dimensions
    via Gaussian copula.
    https://arxiv.org/abs/1504.04093v1
    """

    def __init__(self, corr=None, cov=None, marginals=None, marginal_samples=None):
        self.marginals = marginals or estimate_densities(marginal_samples)
        self._handle_corr(corr, cov)

    def _handle_corr(self, corr, cov):
        corrp = corr is not None
        covp = cov is not None
        {(False, False): _raise(ValueError("Must provide either a covariance or a correlation matrix.")),
         (True, False): self._handle_corr1,
         (False, True): self._handle_corr2,
         (True, True): self._handle_corr3}.get((corrp, covp))(corr, cov)

    def _handle_corr1(self, corr, cov):
        self.corr = corr

    def _handle_corr2(self, corr, cov):
        self.cov = cov
        self.corr = cov2corr(cov)

    def _handle_corr3(self, corr, cov):
        self.corr = corr
        self.cov = cov

    def logpdf(self, theta):
        """Evaluate the logarithm of the density function of the meta-Gaussian distribution.

        Parameters
        ----------
        theta : np.ndarray
            the evaluation point

        See Also
        --------
        pdf
        """
        if len(theta.shape) == 1:
            return self._logpdf(theta)
        elif len(theta.shape) == 2:
            return np.array([self._logpdf(t) for t in theta])

    def pdf(self, theta):
        """Evaluate the probability density function of the meta-Gaussian distribution.

        The probability density function is given by
        .. math::
            g(\theta) = \frac{1}{|\Lambda|^{\frac12}}
            \exp\left{ \frac{1}{2}\eta^{T}(I - \Lambda^{-1})\eta \right}
            \prod_{i=1}^{p} g_i(\theta_i),

        where :math:`\eta` is multivariate normal :math:`\eta \sim N(0, \Lambda)` and :math`g_i`
        are the marginal density functions.

        Parameters
        ----------
        theta : np.ndarray
            the evaluation point

        See Also
        --------
        logpdf
        """
        return np.exp(self.logpdf(theta))

    def _marginal_prod(self, theta):
        """Evaluate the logarithm of the product of the marginals."""
        res = 0
        for (i, t) in enumerate(theta):
            res += self.marginals[i].logpdf(t)
        return res

    def _eta_i(self, i, t):
        return ss.norm.ppf(self.marginals[i].cdf(t))

    def _eta(self, theta):
        return np.array([self._eta_i(i, t) for (i, t) in enumerate(theta)])

    def _logpdf(self, theta):
        correlation_matrix = self.corr
        n, m = correlation_matrix.shape
        a = np.log(1/np.sqrt(np.linalg.det(correlation_matrix)))
        L = np.eye(n) - np.linalg.inv(correlation_matrix)
        quadratic = 1/2 * self._eta(theta).T.dot(L).dot(self._eta(theta))
        c = self._marginal_prod(theta)
        return a + quadratic + c

    def _plot_marginal(self, inx, bounds, points=100):
        import matplotlib.pyplot as plt
        t = np.linspace(*bounds, points)
        return plt.plot(t, self.marginals[inx].pdf(t))

    def rvs(self, n):
        """Sample values from the distribution.

        Parameters
        ----------
        n : int
            The number of samples to produce
        """
        # FIXME: What to do when the meta-Gaussian is initialized with a correlation matrix?
        X = ss.multivariate_normal.rvs(cov=self.cov, size=n)
        U = ss.norm.cdf(X)
        return np.array([m.ppf(U[:, i]) for (i, m) in enumerate(self.marginals)]).T

    __call__ = logpdf


def _set(x):
    try:
        return set(x)
    except TypeError:
        return set([x])


def _concat_ind(x, y):
    return _set(x).union(_set(y))


def make_union(informative_indices):
    """Construct the indicators for the pairwise summary statistics.

    Parameters
    ----------
    informative_indices
    """
    res = informative_indices.copy()
    univariate = filter(lambda p: isinstance(p, int), informative_indices)
    # TODO: symmetric
    pairs = itertools.combinations(univariate, 2)
    for pair in pairs:
        if pair not in res:
            i, j = pair
            res[pair] = _concat_ind(res[i], res[j])

    return res


def sliced_summary(indices):
    """A closure to return specific indices from a summary statistic.

    Parameters
    ----------
    indices : set
      a set of indices

    Returns
    -------
    summary
      a function which slices into an array
    """
    indices = _set(indices)

    def summary(data):
        return data[:, sorted(indices)]
    return summary


def make_distances(full_indices, simulator):
    """Construct a summary statistic and a discrepancy node for each set of indices.

    Parameters
    ----------
    full_indices : dict
      a dictionary specifying the informative summary statistics for each parameter
    simulator : elfi.Simulator
      the simulator
    """
    res = {}
    for i, pair in enumerate(full_indices.items()):
        param, indices = pair
        summary = elfi.Summary(sliced_summary(indices), simulator, name='S{}'.format(i))
        res[param] = elfi.Distance('euclidean', summary, name='D{}'.format(i))

    return res


def make_samplers(dist_dict, sampler_class, **kwargs):
    """Construct samplers.

    Parameters
    ----------
    dist_dict
    sampler_class
      the class of the sampler to use (for example elfi.Rejection)
    **kwargs
      arguments to pass to the sampler
    """
    res = {}
    for k, dist in dist_dict.items():
        res[k] = sampler_class(dist, **kwargs)

    return res


def get_samples(marginal, samplers, parameter, n_samples, **kwargs):
    """Sample from a marginal distribution.

    Parameters
    ----------
    marginal : int or tuple
    samplers : dict
    parameter : str
    n_samples : int
    **kwargs
      additional arguments for sampling
    """
    return samplers[marginal].sample(n_samples, **kwargs).outputs[parameter][:, marginal]


def _full_cor_matrix(correlations, n):
    """Construct a full correlation matrix from pairwise correlations."""
    I = np.eye(n)
    O = np.zeros((n, n))
    indices = itertools.combinations(range(n), 2)
    for (i, inx) in enumerate(indices):
        O[inx] = correlations[i]

    # symmetrize
    return O + O.T + I


def estimate_correlation(marginal, samplers, parameter, n_samples, **kwargs):
    """Estimate an entry in hte correlation matrix.

    Parameters
    ----------
    marginal : tuple
    samplers : dict
    parameter : str
    n_samples : int
    **kwargs
      additional arguments for sampling
    """
    samples = get_samples(marginal, samplers=samplers, parameter=parameter,
                          n_samples=n_samples, **kwargs)
    c1, c2 = samples[:, 0], samples[:, 1]
    r1 = ss.rankdata(c1)
    r2 = ss.rankdata(c2)
    eta1 = ss.norm.ppf(r1/(n_samples + 1))
    eta2 = ss.norm.ppf(r2/(n_samples + 1))
    r, p_val = ss.pearsonr(eta1, eta2)
    return r


def estimate_correlation_matrix(dim, samplers, parameter, n_samples, **kwargs):
    """Construct an estimated correlation matrix.

    Parameters
    ----------
    dim : int
    samplers : dict
    n_samples : int
    **kwargs
      additional arguments for sampling
    """
    pairs = itertools.combinations(range(dim), 2)
    correlations = [estimate_correlation(marginal=marginal,
                                         samplers=samplers, parameter=parameter,
                                         n_samples=n_samples, **kwargs) for marginal in pairs]
    cor = _full_cor_matrix(correlations, dim)
    return cor


def estimate_marginal_density(marginal, samplers, parameter, n_samples, **kwargs):
    """Estimate a univariate marginal probability density function.

    Parameters
    ----------
    marginal
    samplers
    parameter : str
    n_samples : int
    """
    return EmpiricalDensity(get_samples(marginal, samplers=samplers,
                                        parameter=parameter, n_samples=n_samples,  **kwargs))


def estimate_marginals(samplers, parameter, n_samples, **kwargs):
    """Estimate all the univariate marginal probability density functions.

    Parameters
    ----------
    samplers
    parameter : str
    n_samples : int
      the number of samples
    **kwargs
      additional arguments for sampling
    """
    univariate = filter(lambda p: isinstance(p, int), samplers)
    return [EmpiricalDensity(get_samples(u, samplers=samplers, parameter=parameter,
                                         n_samples=n_samples, **kwargs))
            for u in univariate]


def estimate(informative_summaries, simulator, parameter, n_samples=100, sampler_kwargs=None, **kwargs):
    """Perform the Copula ABC estimation.

    Parameters
    ----------
    informative_summaries
    simulator
    n_samples
    sampler_kwargs : dict
    **kwargs
      additional arguments for sampling
    """
    simulator_name = simulator.name
    model = simulator.model.copy()
    simulator = model.get_reference(simulator_name)

    dim = len(list(filter(lambda p: isinstance(p, int), informative_summaries)))  # TODO: use list comp
    und = make_union(informative_summaries)
    dis = make_distances(und, simulator)
    samp = make_samplers(dis, elfi.Rejection, **sampler_kwargs)
    emp = estimate_marginals(samplers=samp, parameter=parameter, n_samples=n_samples, **kwargs)
    cm = estimate_correlation_matrix(dim, samplers=samp, parameter=parameter, n_samples=n_samples, **kwargs)

    return MetaGaussian(corr=cm, marginals=emp)


# class CopulaABC(object):
#     def __init__(self, sampler_class):
#         self.metagaussian = None
#         # self.samplers = samplers

#     def estimate(self, informative_summaries, simulator, n_samples, samplerkwargs, **kwargs):
#         simulator_name = simulator.name
#         model = simulator.model.copy()
#         simulator = model.get_reference(simulator_name)

#         self.metagaussian = estimate(informative_summaries=informative_summaries,
#                                      simulator=simulator, n_samples=n_samples,
#                                      samplerkwargs=samplerkwargs, **kwargs)
