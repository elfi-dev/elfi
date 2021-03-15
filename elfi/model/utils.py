"""Utilities for ElfiModels."""

import numpy as np

import elfi.model.augmenter as augmenter
from elfi.clients.native import Client
from elfi.model.elfi_model import ComputationContext


def rvs_from_distribution(*params, batch_size, distribution, size=None, random_state=None):
    """Transform the rvs method of a scipy like distribution to an operation in ELFI.

    Parameters
    ----------
    params :
        Parameters for the distribution
    batch_size : number of samples
    distribution : scipy-like distribution object
    size : tuple
        Size of a single datum from the distribution.
    random_state : RandomState object or None

    Returns
    -------
    random variates from the distribution

    Notes
    -----
    Used internally by the RandomVariable to wrap distributions for the framework.

    """
    if size is None:
        size = (batch_size, )
    else:
        size = (batch_size, ) + size

    rvs = distribution.rvs(*params, size=size, random_state=random_state)
    return rvs


def distance_as_discrepancy(dist, *summaries, observed):
    """Evaluate a distance function with signature `dist(summaries, observed)` in ELFI."""
    summaries = np.column_stack(summaries)
    # Ensure observed are 2d
    observed = np.concatenate([np.atleast_2d(o) for o in observed], axis=1)
    try:
        d = dist(summaries, observed)
    except ValueError as e:
        raise ValueError('Incompatible data shape for the distance node. Please check '
                         'summary (XA) and observed (XB) output data dimensions. They '
                         'have to be at most 2d. Especially ensure that summary nodes '
                         'outputs 2d data even with batch_size=1. Original error message '
                         'was: {}'.format(e))
    if d.ndim == 2 and d.shape[1] == 1:
        d = d.reshape(-1)
    return d


# TODO: check that there are no latent variables in parameter parents.
#       pdfs and gradients wouldn't be correct in those cases as it would require
#       integrating out those latent variables. This is equivalent to that all
#       stochastic nodes are parameters.
# TODO: could use some optimization
# TODO: support the case where some priors are multidimensional
class ModelPrior:
    """Construct a joint prior distribution over all the parameter nodes in `ElfiModel`."""

    def __init__(self, model):
        """Initialize a ModelPrior.

        Parameters
        ----------
        model : ElfiModel

        """
        model = model.copy()
        self.parameter_names = model.parameter_names
        self.dim = len(self.parameter_names)
        self.client = Client()

        # Prepare nets for the pdf methods
        self._pdf_node = augmenter.add_pdf_nodes(model, log=False)[0]
        self._logpdf_node = augmenter.add_pdf_nodes(model, log=True)[0]

        self._rvs_net = self.client.compile(model.source_net, outputs=self.parameter_names)
        self._pdf_net = self.client.compile(model.source_net, outputs=self._pdf_node)
        self._logpdf_net = self.client.compile(model.source_net, outputs=self._logpdf_node)

    def rvs(self, size=None, random_state=None):
        """Sample the joint prior."""
        random_state = np.random if random_state is None else random_state

        context = ComputationContext(size or 1, seed='global')
        loaded_net = self.client.load_data(self._rvs_net, context, batch_index=0)

        # Change to the correct random_state instance
        # TODO: allow passing random_state to ComputationContext seed
        loaded_net.nodes['_random_state'].update({'output': random_state})
        del loaded_net.nodes['_random_state']['operation']

        batch = self.client.compute(loaded_net)
        rvs = np.column_stack([batch[p] for p in self.parameter_names])

        if self.dim == 1:
            rvs = rvs.reshape(size or 1)

        return rvs[0] if size is None else rvs

    def pdf(self, x):
        """Return the density of the joint prior at x."""
        return self._evaluate_pdf(x)

    def logpdf(self, x):
        """Return the log density of the joint prior at x."""
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

        # TODO: we could add a seed value that would load a "random state" instance
        #       throwing an error if it is used, for instance seed="not used".
        context = ComputationContext(len(x), seed=0)
        loaded_net = self.client.load_data(net, context, batch_index=0)

        # Override
        for k, v in batch.items():
            loaded_net.nodes[k].update({'output': v})
            del loaded_net.nodes[k]['operation']

        val = self.client.compute(loaded_net)[node]
        if ndim == 0 or (ndim == 1 and self.dim > 1):
            val = val[0]

        return val

    def gradient_pdf(self, x):
        """Return the gradient of density of the joint prior at x."""
        raise NotImplementedError

    def gradient_logpdf(self, x, stepsize=None):
        """Return the gradient of log density of the joint prior at x.

        Parameters
        ----------
        x : float or np.ndarray
        stepsize : float or list
            Stepsize or stepsizes for the dimensions

        """
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        grads = np.zeros_like(x)

        for i in range(len(grads)):
            xi = x[i]
            grads[i] = numgrad(self.logpdf, xi, h=stepsize)

        grads[np.isinf(grads)] = 0
        grads[np.isnan(grads)] = 0

        if ndim == 0 or (ndim == 1 and self.dim > 1):
            grads = grads[0]
        return grads

    def _to_batch(self, x):
        return {p: x[:, i] for i, p in enumerate(self.parameter_names)}
