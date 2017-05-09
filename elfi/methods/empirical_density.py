import numpy as np
import scipy.stats as ss
import scipy.optimize as opt


class EmpiricalDensity(object):
    """A wrapper for a Gaussian kernel density estimate.
    
    Parameters
    ----------
    samples : np.ndarray
        a univariate sample

    Attributes
    ----------
    kde :
        a Gaussian kernel density estimate
    support : tuple
        an approximate support for the empirical distribution
    """

    def __init__(self, samples, **kwargs):
        self.kde = ss.gaussian_kde(samples, **kwargs)
        # Try cover the support of the pdf
        # TODO: Use the kde
        a, b = (min(samples), max(samples))
        extension = (b - a)*0.5
        self.support = (a - extension, b + extension)
        # self.pdf.__doc__ = self.kde.pdf.__doc__
        # self.logpdf.__doc__ = self.kde.logpdf.__doc__

    @property
    def dataset(self):
        return self.kde.dataset
    
    @property
    def n(self):
        return self.kde.n
        
    def pdf(self, x):
        return self.kde.pdf(x)

    def logpdf(self, x):
        return self.kde.logpdf(x)

    def _cdf(self, x):
        return self.kde.integrate_box_1d(-np.inf, x)

    def cdf(self, x):
        """Cumulative distribution function of the empirical distribution.

        Parameters
        ----------
        x : array_like
            quantiles
        """
        if isinstance(x, np.ndarray):
            return np.array([self._cdf(xi) for xi in x.flat])
        else:
            return self._cdf(x)

    def _ppf(self, q):
        return opt.brentq(lambda x: self._cdf(x) - q, *self.support)

    def ppf(self, q):
        """Percent point function (inverse of `cdf`) at q
        of the empirical distribution.

        Parameters
        ----------
        q : array_like
            lower tail probability
        """
        if isinstance(q, np.ndarray):
            return np.array([self._ppf(qi) for qi in q.flat])
        else:
            return self._ppf(q)

    def rvs(self, n):
        """Sample n values from the empirical density."""
        u = np.random.rand(n)
        return self.ppf(u)


def estimate_densities(marginal_samples, **kwargs):
    """Compute Gaussian kernel density estimates.

    Parameters
    ----------
    marginal_samples : np.ndarray
        a NxM array of N observations in M variables
    **kwargs :
        additional arguments
    
    Returns
    -------
    empirical_densities : 
       a list of EmpiricalDensity objects
    """
    return [EmpiricalDensity(marginal_samples[:, i], **kwargs)
            for i in range(marginal_samples.shape[1])]
