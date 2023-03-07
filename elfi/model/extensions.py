"""Extensions: ScipyLikeDistribution."""

import warnings

import numpy as np

import elfi.model.augmenter as augmenter
from elfi.clients.native import Client
from elfi.methods.utils import numgrad
from elfi.model.elfi_model import ComputationContext


# TODO: move somewhere else?
class ScipyLikeDistribution:
    """Abstract class for an ELFI compatible random distribution.

    You can implement this as having all methods as classmethods or making an instance.
    Hence the signatures include this, instead of self or cls.

    Note that the class signature is a subset of that of `scipy.rv_continuous`.

    Additionally, methods like BOLFI require information about the gradient of logpdf.
    You can implement this as a classmethod `gradient_logpdf` with the same call signature
    as `logpdf`, and return type of np.array. If this is unimplemented, ELFI will
    approximate it numerically.
    """

    def __init__(self, name=None):
        """Constuctor (optional, only if instances are meant to be used).

        Parameters
        ----------
        name : str
            Name of the distribution.

        """
        self._name = name or self.__class__.__name__

    @classmethod
    def rvs(this, *params, size=1, random_state):
        """Generate random variates.

        Parameters
        ----------
        param1, param2, ... : array_like
            Parameter(s) of the distribution
        size : int or tuple of ints, optional
        random_state : RandomState

        Returns
        -------
        rvs : ndarray
            Random variates of given size.

        """
        raise NotImplementedError

    @classmethod
    def pdf(this, x, *params, **kwargs):
        """Probability density function at x.

        Parameters
        ----------
        x : array_like
           points where to evaluate the pdf
        param1, param2, ... : array_like
           parameters of the model

        Returns
        -------
        pdf : ndarray
           Probability density function evaluated at x

        """
        raise NotImplementedError

    @classmethod
    def logpdf(this, x, *params, **kwargs):
        """Log of the probability density function at x.

        Parameters
        ----------
        x : array_like
            Points where to evaluate the logpdf.
        param1, param2, ... : array_like
            Parameters of the model.
        kwargs

        Returns
        -------
        logpdf : ndarray
           Log of the probability density function evaluated at x.

        """
        p = this.pdf(x, *params, **kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ans = np.log(p)

        return ans

    @property
    def name(this):
        """Return the name of the distribution."""
        if hasattr(this, '_name'):
            return this._name
        elif isinstance(this, type):
            return this.__name__
        else:
            return this.__class__.__name__


# TODO: check that there are no latent variables in parameter parents.
#       pdfs and gradients wouldn't be correct in those cases as it would require
#       integrating out those latent variables. This is equivalent to that all
#       stochastic nodes are parameters.
# TODO: could use some optimization
# TODO: support the case where some priors are multidimensional
class ModelPrior:
    """Construct a joint prior distribution over all or selected parameter nodes in `ElfiModel`."""

    def __init__(self, model, parameter_names=None):
        """Initialize a ModelPrior.

        Parameters
        ----------
        model : ElfiModel
        parameter_names : list, optional
            Parameters included in the prior and their order. Default model.parameter_names.

        """
        model = model.copy()

        if parameter_names is None:
            self.parameter_names = model.parameter_names
        elif isinstance(parameter_names, list):
            for param in parameter_names:
                if param not in model.parameter_names:
                    raise ValueError(f"Parameter \'{param}\' not found in model parameters.")
            self.parameter_names = parameter_names
        else:
            raise ValueError("parameter_names must be a list of strings.")

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
