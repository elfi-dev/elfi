"""This module implements empirical estimations of distributions.

References
----------
Jingjing Li, David J. Nott, Yanan Fan, Scott A. Sisson
Extending approximate Bayesian computation methods to high dimensions via Gaussian copula.
Computational Statistics & Data Analysis
Volume 106, February 2017, Pages 77-89
https://doi.org/10.1016/j.csda.2016.07.005

"""
import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d

__all__ = ('EmpiricalDensity', 'ecdf', 'eppf', 'estimate_densities', 'MetaGaussian')


def ecdf(samples):
    """Compute an empirical cdf.

    Parameters
    ----------
    samples : array_like
      a univariate sample

    Returns
    -------
    empirical_cdf
      an interpolated function for the estimated cdf

    """
    x, y = _handle_endpoints(*_ecdf(samples))
    return _interp_ecdf(x, y)


def eppf(samples):
    """Compute an empirical quantile function.

    Parameters
    ----------
    samples : array_like
      a univariate sample

    Returns
    -------
    empirical_ppf
      an interpolated function for the estimated quantile function

    """
    x, y = _handle_endpoints(*_ecdf(samples))
    return _interp_ppf(x, y)


def _ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1)/float(len(xs))
    return xs, ys


def _low(xs, ys):
    """Compute the interception point (x, 0)."""
    slope = (ys[1] - ys[0])/(xs[1] - xs[0])
    intersect = -ys[0]/slope + xs[0]
    return intersect


def _high(xs, ys):
    """Compute the interception point (x, 1)."""
    slope = (ys[-1] - ys[-2])/(xs[-1] - xs[-2])
    intersect = (1 - ys[-1])/slope + xs[-1]
    return intersect


def _handle_endpoints(xs, ys):
    high = _high(xs, ys)
    low = _low(xs, ys)

    # add endpoints
    x = np.append(np.insert(xs, 0, low), high)
    y = np.append(np.insert(ys, 0, 0.), 1.)
    return x, y


def _interp_ecdf(x, y, **kwargs):
    low, high = x[0], x[-1]

    # linear interpolation
    f = interp1d(x, y, **kwargs)

    def interp(q):
        if isinstance(q, np.ndarray):
            too_low = sum(q < low)
            too_high = sum(q > high)
            return np.concatenate([np.zeros(too_low),
                                   f(q[(q >= low) & (q <= high)]),
                                   np.ones(too_high)])
        else:
            return _scalar_cdf(f, low, high, q)

    return interp


def _scalar_cdf(f, low, high, q):
    if q < low:
        return 0
    elif q > high:
        return 1
    else:
        return f(q)


def _interp_ppf(x, y, **kwargs):
    f = interp1d(y, x, **kwargs)

    def interp(p):
        try:
            return f(p)
        except ValueError:
            raise ValueError("The quantile function is not defined outside [0, 1].")

    return interp


class EmpiricalDensity(object):
    """An empirical approximation of a random variable.

    The density function is approximated using the gaussian
    kernel density estimation from scipy (scipy.stats.gaussian_kde).
    The cumulative distribution function and quantile function are constructed
    the linearly interpolated empirical cumulative distribution function.

    Attributes
    ----------
    kde :
        a Gaussian kernel density estimate

    """

    def __init__(self, samples, **kwargs):
        """Create an empirical estimation of a distribution.

        Parameters
        ----------
        samples : np.ndarray
            a univariate sample
        **kwargs
            additional arguments for kernel density estimation

        """
        self.kde = ss.gaussian_kde(samples, **kwargs)
        x, y = _handle_endpoints(*_ecdf(samples))
        self.cdf = _interp_ecdf(x, y)
        self.ppf = _interp_ppf(x, y)

    @property
    def dataset(self):
        """Get the dataset used for fitting the kernel density estimate."""
        return self.kde.dataset

    @property
    def n(self):
        """Get the number of samples used for the kernel density estimation."""
        return self.kde.n

    def pdf(self, x):
        """Compute the estimated pdf."""
        return self.kde.pdf(x)

    def logpdf(self, x):
        """Compute the estimated logarithmic pdf."""
        return self.kde.logpdf(x)

    def rvs(self, size=1, random_state=None, approximation=False,
            replace=True, p=None):
        """Sample values from the empirical density.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is a tuple ``(m, n, k)``,
            then ``m*n*k`` samples are drawn. If approximation is set to true,
            only integer values are allowed. The default size is one.
        random_state : numpy.random.RandomState, optional
            An instance of `numpy.random.RandomState`. Can be used to produce
            reproducible results. By default the results are stochastic.
        approximation : boolean, optional
            Whether to use inverse transform sampling or to sample from the dataset.
            The default is `False`, which means that sampling is done from the dataset.
        replace : boolean, optional
            When sampling from the dataset, should the sample be with
            or without replacement?
        p : 1-D array-like, optional
            The probabilities associated with each entry in the dataset.
            The default is a uniform distribution over the entries. This is only
            a valid option when approximation is set to `False`.

        Returns
        -------
        samples : single item or ndarray
            The generated random samples.

        """
        rs = random_state or np.random.RandomState()
        if approximation:
            assert isinstance(size, int), "The size should be an integer."
            return self.ppf(rs.rand(size))
        else:
            return rs.choice(a=self.dataset, size=size, replace=replace, p=p)

    @classmethod
    def name(cls):
        """Get the name of this distribution."""
        return cls.__class__.__name__


def estimate_densities(marginal_samples, **kwargs):
    """Compute empirical estimates for distributions.

    Parameters
    ----------
    marginal_samples : np.ndarray
        a NxM array of N observations in M variables
    **kwargs :
        additional arguments for kernel density estimation

    Returns
    -------
    empirical_densities :
       a list of EmpiricalDensity objects

    """
    return [EmpiricalDensity(marginal_samples[:, i], **kwargs)
            for i in range(marginal_samples.shape[1])]


def _raise(err):
    """Exception raising closure."""
    def fun():
        raise err
    return fun


class MetaGaussian(object):
    """A meta-Gaussian distribution.

    Attributes
    ----------
    corr : np.ndarray
        The correlation matrix of the meta-Gaussian distribution.
    marginals : List
        A list of objects that implement 'cdf' and 'ppf' methods.
    dim : int
        The number of dimensions.

    References
    ----------
    Jingjing Li, David J. Nott, Yanan Fan, Scott A. Sisson
    Extending approximate Bayesian computation methods to high dimensions via Gaussian copula.
    Computational Statistics & Data Analysis
    Volume 106, February 2017, Pages 77-89
    https://doi.org/10.1016/j.csda.2016.07.005

    """

    def __init__(self, corr, marginals=None, marginal_samples=None):
        """Create a meta-Gaussian distribution.

        Parameters
        ----------
        corr : np.ndarray
            Tha correlation matrix of the meta-Gaussian distribution.
        marginals : density_like
            A list of objects that implement 'cdf' and 'ppf' methods.
        marginal_samples : np.ndarray
            A NxM array of samples, where N is the number of observations
            and m is the number of dimensions.

        """
        self._handle_marginals(marginals, marginal_samples)
        self.corr = corr
        self.dim = len(marginals)

    @classmethod
    def name(cls):
        """Get the name of this distribution."""
        return cls.__class__.__name__

    def _handle_marginals(self, marginals, marginal_samples):
        marginalp = marginals is not None
        marginal_samplesp = marginal_samples is not None
        {(False, False): _raise(ValueError("Must provide either marginals or marginal_samples.")),
         (True, False): self._handle_marginals1,
         (False, True): self._handle_marginals2,
         (True, True): self._handle_marginals3}\
         .get((marginalp, marginal_samplesp))(marginals, marginal_samples)  # noqa

    def _handle_marginals1(self, marginals, marginal_samples):
        self.marginals = marginals

    def _handle_marginals2(self, marginals, marginal_samples):
        self.marginals = estimate_densities(marginal_samples)

    def _handle_marginals3(self, marginals, marginal_samples):
        self.marginals = marginals

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
        r"""Evaluate the probability density function of the meta-Gaussian distribution.

        The probability density function is given by

        .. math::
            g(\theta) = \frac{1}{|R|^{\frac12}}
                        \exp \left\{-\frac{1}{2} u^T (R^{-1} - I) u \right\}
                        \prod_{i=1}^p g_i(\theta_i) \, ,

        where :math:`\phi` is the standard normal density, :math:`u_i = \Phi^{-1}(G_i(\theta_i))`,
        :math:`g_i` are the marginal densities, and :math:`R` is a correlation matrix.

        Parameters
        ----------
        theta : np.ndarray
            the evaluation point

        See Also
        --------
        logpdf

        """
        return np.exp(self.logpdf(theta))

    def rvs(self, size=1, random_state=None):
        """Sample values from the empirical density.

        Parameters
        ----------
        size : int, optional
            The number of samples to produce. The default size is one.
        random_state : numpy.random.RandomState, optional
            An instance of `numpy.random.RandomState`. Can be used to produce
            reproducible results. By default the results are stochastic.

        Returns
        -------
        samples : ndarray
            The generated random samples.

        """
        Z = ss.multivariate_normal.rvs(cov=self.corr, mean=np.zeros(self.dim),  # noqa
                                       size=size, random_state=random_state)
        U = ss.norm.cdf(Z)  # noqa
        return np.array([m.ppf(U[:, i]) for (i, m) in enumerate(self.marginals)]).T

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
        L = np.eye(n) - np.linalg.inv(correlation_matrix)  # noqa
        quadratic = 1/2 * self._eta(theta).T.dot(L).dot(self._eta(theta))
        c = self._marginal_prod(theta)
        return a + quadratic + c

    def _plot_marginal(self, inx, bounds, points=100):
        import matplotlib.pyplot as plt
        t = np.linspace(*bounds, points)
        return plt.plot(t, self.marginals[inx].pdf(t))

    __call__ = logpdf
