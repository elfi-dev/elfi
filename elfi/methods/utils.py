import logging
import warnings

import numpy as np
import scipy.stats as ss
import numdifftools

from elfi.model.elfi_model import ComputationContext
import elfi.model.augmenter as augmenter
from elfi.clients.native import Client
from elfi.utils import get_sub_seed

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
            scalar, 1d or 2d array of points where to evaluate, observations in rows
        means : array_like
            means of the Gaussian mixture components. It is assumed that means[0] contains
            the mean of the first gaussian component.
        weights : array_like
            1d array of weights of the gaussian mixture components
        cov : array_like
            a shared covariance matrix for the mixture components
        """

        means, weights = cls._normalize_params(means, weights)

        ndim = np.asanyarray(x).ndim
        if means.ndim == 1:
            x = np.atleast_1d(x)
        if means.ndim == 2:
            x = np.atleast_2d(x)

        d = np.zeros(len(x))
        for m, w in zip(means, weights):
            d += w * ss.multivariate_normal.pdf(x, mean=m, cov=cov)

        # Cast to correct ndim
        if ndim == 0 or (ndim==1 and means.ndim==2):
            return d.squeeze()
        else:
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
        if means.ndim > 2:
            raise ValueError('means.ndim = {} but must be at most 2.'.format(means.ndim))

        if weights is None:
            weights = np.ones(len(means))
        weights = normalize_weights(weights)
        return means, weights


def numgrad(fn, x, h=0.00001):
    """

    Parameters
    ----------
    fn
    x : np.ndarray
        A single point in 1d vector
    h

    Returns
    -------

    """

    x = np.atleast_1d(x)
    x = np.column_stack((x - h, x, x + h))
    dim = len(x)

    # This creates some unnecessary computations, you only need to vary one dimension at a
    # time
    mgrid = np.meshgrid(*x)
    shape = mgrid[0].shape
    xgrid = np.column_stack(tuple([param.reshape(-1) for param in mgrid]))

    f = fn(xgrid)
    f = f.reshape(shape)

    fgrad = np.gradient(f, h)
    if dim > 1:
        take = (1,) * dim
        grad = np.array([fg[take] for fg in fgrad])

        # Make yourself clear why the first two are reversed
        swap = grad[0]
        grad[0] = grad[1]
        grad[1] = swap
    else:
        grad = fgrad[1]

    return grad


# TODO: check that there are no latent variables in parameter parents.
#       pdfs and gradients wouldn't be correct in those cases as it would require integrating out those latent
#       variables. This is equivalent to that all stochastic nodes are parameters.
# TODO: needs some optimization
class ModelPrior:
    """Constructs a joint prior distribution for all the parameter nodes in `ElfiModel`"""

    def __init__(self, model):
        model = model.copy()
        self.parameters = model.parameters
        self.dim = len(self.parameters)
        self.client = Client()

        self.context = ComputationContext()

        # Prepare nets for the pdf methods
        self._pdf_node = augmenter.add_pdf_nodes(model, log=False)[0]
        self._logpdf_node = augmenter.add_pdf_nodes(model, log=True)[0]

        self._rvs_net = self.client.compile(model.source_net, outputs=self.parameters)
        self._pdf_net = self.client.compile(model.source_net, outputs=self._pdf_node)
        self._logpdf_net = self.client.compile(model.source_net, outputs=self._logpdf_node)

    def rvs(self, size=None, random_state=None):
        random_state = random_state or np.random

        self.context.batch_size = size or 1
        self.context.seed = get_sub_seed(random_state, 0)

        loaded_net = self.client.load_data(self._rvs_net, self.context, batch_index=0)
        batch = self.client.compute(loaded_net)
        rvs = np.column_stack([batch[p] for p in self.parameters])

        if self.dim == 1:
            rvs = rvs.reshape(size or 1)

        return rvs[0] if size is None else rvs

    def pdf(self, x):
        return self._evaluate_pdf(x)

    def logpdf(self, x):
        return self._evaluate_pdf(x, log=True)

    def _evaluate_pdf(self, x, log=False):
        if log:
            net = self._logpdf_net
            node = self._logpdf_node
        else:
            net = self._pdf_net
            node = self._pdf_node

        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))
        batch = self._to_batch(x)

        self.context.batch_size = len(x)
        loaded_net = self.client.load_data(net, self.context, batch_index=0)

        # Override
        for k, v in batch.items(): loaded_net.node[k] = {'output': v}

        val = self.client.compute(loaded_net)[node]
        if ndim == 0 or (ndim==1 and self.dim > 1):
            val = val[0]

        return val

    def gradient_pdf(self, x):
        raise NotImplementedError

    def gradient_logpdf(self, x):
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        grads = np.zeros_like(x)

        for i in range(len(grads)):
            xi = x[i]
            grads[i] = numgrad(self.logpdf, xi)

        grads[np.isinf(grads)] = 0
        grads[np.isnan(grads)] = 0

        if ndim == 0 or (ndim==1 and self.dim > 1):
            grads = grads[0]
        return grads

    def _to_batch(self, x):
        return {p: x[:, i] for i, p in enumerate(self.parameters)}
