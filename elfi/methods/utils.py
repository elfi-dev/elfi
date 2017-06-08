import logging

import numpy as np
import scipy.stats as ss
import numdifftools

from elfi.model.elfi_model import ComputationContext
import elfi.model.augmenter as augmenter
from elfi.clients.native import Client

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


# TODO: check that there are no latent variables in parameter parents.
#       pdfs and gradients wouldn't be correct in those cases as it would require integrating out those latent
#       variables. This is equivalent to that all stochastic nodes are parameters.
class ModelPrior:
    """Constructs a joint prior distribution for all the parameter nodes in `ElfiModel`"""

    def __init__(self, model):
        model = model.copy()
        self.parameters = model.parameters
        self.client = Client()

        self.context = ComputationContext()

        # Prepare nets for the pdf methods
        self._pdf_node = augmenter.add_pdf_nodes(model, log=False)[0]
        self._logpdf_node = augmenter.add_pdf_nodes(model, log=True)[0]

        self._rvs_net = self.client.compile(model.source_net, outputs=self.parameters)
        self._pdf_net = self.client.compile(model.source_net, outputs=self._pdf_node)
        self._logpdf_net = self.client.compile(model.source_net, outputs=self._logpdf_node)

    # TODO: add random_state
    def rvs(self, size=1):
        self.context.batch_size = size
        loaded_net = self.client.load_data(self._rvs_net, self.context, batch_index=0)
        batch = self.client.compute(loaded_net)
        return np.column_stack([batch[p] for p in self.parameters])

    def pdf(self, x):
        x = np.atleast_2d(x)
        batch = self._to_batch(x)

        self.context.batch_size = len(x)
        loaded_net = self.client.load_data(self._pdf_net, self.context, batch_index=0)

        # Override
        for k, v in batch.items(): loaded_net.node[k] = {'output': v}

        return self.client.compute(loaded_net)[self._pdf_node]

    def logpdf(self, x):
        x = np.atleast_2d(x)
        batch = self._to_batch(x)

        self.context.batch_size = len(x)
        loaded_net = self.client.load_data(self._logpdf_net, self.context, batch_index=0)

        # Override
        for k, v in batch.items(): loaded_net.node[k] = {'output': v}

        return self.client.compute(loaded_net)[self._logpdf_node]

    def gradient_pdf(self, x):
        raise NotImplementedError

    def gradient_logpdf(self, x):
        x = np.atleast_2d(x)
        return numdifftools.Gradient(self.logpdf)(x)

    def _to_batch(self, x):
        return {p: x[:, i] for i, p in enumerate(self.parameters)}

    #def _to_array(self, batch, nodes):
    #    return np.column_stack([batch[p] for p in nodes])

    # TODO: remove, not used
    def numerical_gradient(x, *params, distribution=None, **kwargs):
        """Gradient of the log of the probability density function at x.

        Approximated numerically.

        Parameters
        ----------
        x : array_like
            points where to evaluate the gradient
        param1, param2, ... : array_like
            parameters of the model
        distribution : ScipyLikeDistribution or a distribution from Scipy

        Returns
        -------
        grad_logpdf : ndarray
           Gradient of the log of the probability density function evaluated at x
        """

        # due to the common scenario of logpdf(x) = -inf, multiple confusing warnings could be generated
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')
            if np.isinf(distribution.logpdf(x, *params, **kwargs)):
                grad = np.zeros_like(x)  # logpdf = -inf => grad = 0
            else:
                grad = numdifftools.Gradient(distribution.logpdf)(x, *params, **kwargs)
                grad = np.where(np.isnan(grad), 0, grad)

        return grad
