import itertools

import numpy as np
import scipy.stats as ss

import elfi
from elfi.methods.utils import cov2corr
from .empirical_density import estimate_densities, EmpiricalDensity

def _raise(err):
    def fun():
        raise err
    return fun

class GaussianCopula(object):
    """Gaussian copula estimate of a multivariate distribution.

    This class provides way to estimate multivariate probability
    distributions by separately estimating the marginal distributions
    and the correlations between them.

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
        """Compute the logarithm of the estimated probability density.

        Parameters
        ----------
        theta : np.ndarray
            the evaluation location

        See Also
        --------
        pdf
        """
        if len(theta.shape) == 1:
            return self._logpdf(theta)
        elif len(theta.shape) == 2:
            return np.array([self._logpdf(t) for t in theta])

    def pdf(self, theta):
        """Compute the pdf of the estimated meta-Gaussian distribution.

        The probability density function is given by
        .. math::
            g(\theta) = \frac{1}{|\Lambda|^{\frac12}}
            \exp\left{ \frac{1}{2}\eta^{T}(I - \Lambda^{-1})\eta \right}
            \prod_{i=1}^{p} g_i(\theta_i),

        where :math:`\eta` is multivariate normal :math:`\eta \sim N(0, \Lambda)`.

        Parameters
        ----------
        theta : np.ndarray

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
        """Sample values from the estimated distribution.

        Parameters
        ----------
        n : int
            The number of samples to produce
        """
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

def make_union(p_ind):
    res = p_ind.copy()
    univariate = filter(lambda p: isinstance(p, int), p_ind)
    # TODO: symmetric
    pairs = itertools.combinations(univariate, 2)
    for pair in pairs:
        if pair not in res:
            i, j = pair
            res[pair] = _concat_ind(res[i], res[j])

    return res

def info_ss(indices):
    indices = _set(indices)

    def summary(data):
        return data[:, sorted(indices)]
    return summary

def make_distances(param_ss, simulator):
    res = {}
    for i, pair in enumerate(param_ss.items()):
        param, indices = pair
        summary = elfi.Summary(info_ss(indices), simulator, name='S{}'.format(i))
        res[param] = elfi.Distance('euclidean', summary, name='D{}'.format(i))

    return res

def make_samplers(dist_dict, method_class, **kwargs):
    res = {}
    for k, dist in dist_dict.items():
        res[k] = method_class(dist, **kwargs)

    return res

def get_samples(inx, samplers, n_samples=10, parameter='mu'):
    return samplers[inx].sample(n_samples).outputs[parameter][:, inx]

def _full_cor_matrix(correlations, n):
    """Construct a full correlation matrix from pairwise correlations."""
    I = np.eye(n)
    O = np.zeros((n, n))
    indices = itertools.combinations(range(n), 2)
    for (i, inx) in enumerate(indices):
        O[inx] = correlations[i]

    return O + O.T + I

def _estimate_correlation(marginal, samplers, n_samples):
    samples = get_samples(marginal, samplers=samplers, n_samples=n_samples)
    c1, c2 = samples[:, 0], samples[:, 1]
    r1 = np.argsort(c1) + 1
    r2 = np.argsort(c2) + 1
    n = len(r1) # n_samples?
    eta1 = ss.norm.ppf(r1/(n + 1))
    eta2 = ss.norm.ppf(r2/(n + 1))
    r, p_val = ss.pearsonr(eta1, eta2)
    return r

def _cor_matrix(dim, samplers, n_samples):
    """Construct an estimated correlation matrix."""
    pairs = itertools.combinations(range(dim), 2)
    correlations = [_estimate_correlation(marginal, samplers, n_samples) for marginal in pairs]
    cor = _full_cor_matrix(correlations, dim)
    return cor


def _estimate_marginals(samplers, n_samples):
    univariate = filter(lambda p: isinstance(p, int), samplers)
    return [EmpiricalDensity(get_samples(u, samplers=samplers, n_samples=n_samples))
            for u in univariate]

def estimate(informative_summaries, simulator, n_samples=100, **kwargs):
    dim = len(list(filter(lambda p: isinstance(p, int), informative_summaries)))  # TODO: use list comp
    und = make_union(informative_summaries)
    dis = make_distances(und, simulator)
    samp = make_samplers(dis, elfi.Rejection, **kwargs)
    emp = _estimate_marginals(samp, n_samples=n_samples)
    cm = _cor_matrix(dim, samp, n_samples=n_samples)

    return GaussianCopula(corr=cm, marginals=emp)


# def _full_cor_matrix(correlations, n):
#     """Construct a full correlation matrix from pairwise correlations."""
#     I = np.eye(n)
#     O = np.zeros((n, n))
#     indices = itertools.combinations(range(n), 2)
#     for (i, inx) in enumerate(indices):
#         O[inx] = correlations[i]

#     return O + O.T + I


# def estimate_correlation(indices):
#     """Estimate the correlation between indices (i,j)."""
#     pass


# class CopulaEstimator(object):
#     """Sketch for the copula ABC method from: https://arxiv.org/abs/1504.04093

#     Arguments
#     ---------
#     method: ABCMethod
#       the ABC method used to approximate the marginal distributions
#     parameter_dist: OrderedDict
#       a dictionary that maps parameter nodes to their respective
#       informative discrepancy nodes
#     """

#     def __init__(self, methods1d, methods2d, method=None,
#                  parameter_dist=None, **kwargs):
#         self.method = method
#         self._parameter_dist = parameter_dist
#         self._1d_methods = methods1d
#         self._2d_methods = methods2d

#     def estimate(self, n_samples=100):
#         """Estimate the posterior using a Gaussian copula.

#         Arguments
#         ---------
#         n_samples: int
#           number of samples to use for each marginal estimate
#         """
#         if not self._1d_methods:
#             self._construct_methods()
#         kdes = self._marginal_kdes(n_samples)
#         cm = self._cor_matrix(n_samples)
#         return GaussianCopula(cm, kdes)

#     # def diagnose_quality(self):
#     #     """Evaluate whether the full posterior may be adequately modelled by
#     #     a Gaussian copula."""
#     #     raise NotImplementedError

#     def _construct_methods(self):
#         """Constructs marginal ABC methods with default settings."""
#         #TODO: is it possible to use samples directly instead?
#         self._1d_methods = self._construct_1d_methods()
#         self._2d_methods = self._construct_2d_methods()

#     def _construct_1d_methods(self):
#         methods = {k: self.method(distance_node=v, parameter_nodes=[k])
#                    for k, v in self._parameter_dist.items()}
#         return methods

#     def _construct_2d_methods(self):
#         pairs = itertools.combinations(self._parameter_dist, 2)
#         #TODO: This is only a placeholder
#         methods = {pair: self.method(distance_node=self._parameter_dist[pair[0]],
#                                      parameter_nodes=list(pair))
#                    for pair in pairs}
#         return methods

#     def _sample_from_marginal(self, marginal, n_samples):
#         """Sample from the approximate marginal."""
#         if isinstance(marginal, Prior):
#             return self._sample_1d_marginal(marginal, n_samples)
#         elif isinstance(marginal, tuple):
#             return self._sample_2d_marginal(marginal, n_samples)

#     def _sample_1d_marginal(self, marginal, n_samples):
#         res = self._1d_methods[marginal].sample(n_samples=n_samples)
#         return res.samples[marginal.name]

#     def _sample_2d_marginal(self, marginal, n_samples):
#         res = self._2d_methods[marginal].sample(n_samples=n_samples)
#         sample = res.samples[marginal[0].name], res.samples[marginal[1].name]
#         return sample

#     def _marginal_kde(self, marginal, n_samples):
#         #TODO: add 2d
#         samples = self._sample_from_marginal(marginal, n_samples)
#         kernel = ss.gaussian_kde(samples.reshape(-1))
#         return kernel

#     def _marginal_kdes(self, n_samples):
#         marginal_params = self._parameter_dist
#         kdes = [self._marginal_kde(m, n_samples) for m in marginal_params]
#         return kdes

#     def _estimate_correlation(self, marginal, n_samples):
#         samples = self._sample_from_marginal(marginal, n_samples)
#         c1, c2 = samples[0].reshape(-1), samples[1].reshape(-1)
#         r1 = np.argsort(c1) + 1
#         r2 = np.argsort(c2) + 1
#         n = len(r1)
#         eta1 = ss.norm.ppf(r1/(n + 1))
#         eta2 = ss.norm.ppf(r2/(n + 1))
#         cor = np.corrcoef(eta1, eta2)[0,1]
#         return cor

#     def _cor_matrix(self, n_samples):
#         """Construct an estimated correlation matrix."""
#         pairs = itertools.combinations(self._parameter_dist, 2)
#         correlations = [self._estimate_correlation(marginal, n_samples)
#                         for marginal in pairs]
#         cor = _full_cor_matrix(correlations, self.arity)
#         return cor
