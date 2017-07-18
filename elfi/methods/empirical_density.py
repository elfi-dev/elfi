import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d


def ecdf(samples):
    """Compute an empirical cdf.

    Parameters
    ----------
    samples : array_like
      a univariate sample

    Returns
    -------
    cdf
      an interpolated function for the estimated cdf
    """
    x, y = _handle_endpoints(*_ecdf(samples))
    return _interp_ecdf(x, y)


def ppf(samples):
    """Compute an empirical quantile function.

    Parameters
    ----------
    samples : array_like
      a univariate sample

    Returns
    -------
    ppf
      an interpolated function for the estimated quantile function
    """
    x, y = _handle_endpoints(*_ecdf(samples))
    return _interp_ppf(x, y)


def _ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) +  1)/float(len(xs))
    return xs, ys


def _low(xs, ys):
    """Compute the intercetion point (x, 0)."""
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

    Parameters
    ----------
    samples : np.ndarray
        a univariate sample
    **kwargs
        additional arguments for kernel density estimation

    Attributes
    ----------
    kde :
        a Gaussian kernel density estimate
    """

    def __init__(self, samples, **kwargs):
        self.kde = ss.gaussian_kde(samples, **kwargs)
        x, y = _handle_endpoints(*_ecdf(samples))
        self.cdf = _interp_ecdf(x, y)
        self.ppf = _interp_ppf(x, y)

    @property
    def dataset(self):
        """The dataset used for fitting the kernel density estimate."""
        return self.kde.dataset

    @property
    def n(self):
        """The number of samples used for the kernel density estimation."""
        return self.kde.n

    def pdf(self, x):
        """Compute the estimated pdf."""
        return self.kde.pdf(x)

    def logpdf(self, x):
        """Compute the estimated logarithmic pdf."""
        return self.kde.logpdf(x)

    def rvs(self, n):
        """Sample n values from the empirical density."""
        return self.ppf(np.random.rand(n))


def estimate_densities(marginal_samples, **kwargs):
    """Compute Gaussian kernel density estimates.

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
