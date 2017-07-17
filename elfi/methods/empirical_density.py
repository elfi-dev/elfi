import numpy as np
import scipy.stats as ss
import scipy.optimize as opt
from scipy.interpolate import interp1d


def _ecdf(x):
    """Compute an univariate empirical cdf."""
    xs = np.sort(x)
    ys = np.arange(1, len(xs) +  1)/float(len(xs))
    return xs, ys

def _low(xs, ys):
    """Handle the lower endpoint."""
    slope = (ys[1] - ys[0])/(xs[1] - xs[0])
    intersect = -ys[0]/slope + xs[0]
    return intersect

def _high(xs, ys):
    """Handle the higher endpoint."""
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


def _interp_ecdf(x, y):
    """Compute an interpolated cdf from empirical values."""
    low, high = x[0], x[-1]

    # linear interpolation
    f = interp1d(x, y)

    def interp(x):
        too_low = sum(x < low)
        too_high = sum(x > high)
        return np.concatenate([np.zeros(too_low),
                               f(x[(x >= low) & (x <= high)]),
                               np.ones(too_high)])

    return interp

def _interp_ppf(x, y):
    f = interp1d(y, x)

    def interp(p):
        try:
            return f(p)
        except ValueError:
            raise ValueError("The quantile function is not defined outside [0, 1].")

    return interp


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
        x, y = _handle_endpoints(*_ecdf(samples))
        self.cdf = _interp_ecdf(x, y)
        self.ppf = _interp_ppf(x, y)

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
