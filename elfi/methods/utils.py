import logging

import numpy as np
import scipy.stats as ss


logger = logging.getLogger(__name__)


def normalize_weights(weights):
    w = np.atleast_1d(weights)
    if np.any(w < 0):
        raise ValueError("Weights must be positive")
    wsum = np.sum(weights)
    if wsum == 0:
        raise ValueError("All weights are zero")
    return w/wsum


def weighted_var(x, weights=None):
    """Unbiased weighted variance (sample variance) for the components of x.

    The weights are assumed to be non random (reliability weights).

    Parameters
    ----------
    x : np.ndarray
        1d or 2d with observations in rows
    weights : np.ndarray or None
        1d array of weights. None defaults to standard variance.

    Returns
    -------
    s2 : np.array
        1d vector of component variances

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance

    """
    if weights is None:
        weights = np.ones(len(x))

    V_1 = np.sum(weights)
    V_2 = np.sum(weights**2)

    xbar = np.average(x, weights=weights, axis=0)
    numerator = weights.dot((x - xbar)**2)
    s2 = numerator / (V_1 - (V_2 / V_1))
    return s2


class GMDistribution:
    """Gaussian mixture distribution with a shared covariance matrix."""

    @classmethod
    def pdf(cls, x, means, cov=1, weights=None):
        """Evaluate the density at points x.

        Parameters
        ----------
        x : array_like
            1d or 2d array of points where to evaluate
        means : array_like
            means of the Gaussian mixture components
        weights : array_like
            1d array of weights of the gaussian mixture components
        cov : array_like
            a shared covariance matrix for the mixture components
        """

        x = np.atleast_1d(x)
        means, weights = cls._normalize_params(means, weights)

        d = np.zeros(len(x))
        for m, w in zip(means, weights):
            d += w * ss.multivariate_normal.pdf(x, mean=m, cov=cov)
        return d

    @classmethod
    def rvs(cls, means, cov=1, weights=None, size=1, random_state=None):
        """Random variates from the distribution

        Parameters
        ----------
        x : array_like
            1d or 2d array of points where to evaluate
        means : array_like
            means of the Gaussian mixture components
        weights : array_like
            1d array of weights of the gaussian mixture components
        cov : array_like
            a shared covariance matrix for the mixture components
        size : int or tuple
        random_state : np.random.RandomState or None
        """

        means, weights = cls._normalize_params(means, weights)
        random_state = random_state or np.random

        inds = random_state.choice(len(means), size=size, p=weights)
        rvs = means[inds]
        perturb = ss.multivariate_normal.rvs(mean=means[0]*0,
                                             cov=cov,
                                             random_state=random_state,
                                             size=size)
        return rvs + perturb

    @staticmethod
    def _normalize_params(means, weights):
        means = np.atleast_1d(means)
        if weights is None:
            weights = np.ones(len(means))
        weights = normalize_weights(weights)
        return means, weights


def corr2cov(corr, std):
    """Convert a correlation matrix into a covariance matrix."""
    std = std[:, np.newaxis]
    return std.T * corr * std


def cov2corr(cov):
    """Convert a covariance matrix into a correlation matrix."""
    std = np.sqrt(np.diag(cov))[:, np.newaxis]
    return cov / std.T / std
