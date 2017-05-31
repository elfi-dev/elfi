import logging
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
from functools import partial

import elfi
from elfi.methods.bo.utils import minimize, numerical_grad_logpdf


logger = logging.getLogger(__name__)


class BolfiPosterior(object):
    """
    Container for the approximate posterior in the BOLFI framework, where the likelihood
    is defined as

    L \propto F((h - \mu) / \sigma)

    where F is the cdf of N(0,1), h is a threshold, and \mu and \sigma are the mean and (noisy)
    standard deviation of the Gaussian process.

    Note that when using a log discrepancy, h should become log(h).

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    Parameters
    ----------
    model : object
        Instance of the surrogate model, e.g. elfi.bo.gpy_regression.GPyRegression.
    threshold : float, optional
        The threshold value used in the calculation of the posterior, see the BOLFI paper for details.
        By default, the minimum value of discrepancy estimate mean is used.
    priors : list of elfi.Priors, optional
        By default uniform distribution within model bounds.
    n_inits : int, optional
        Number of initialization points in internal optimization.
    max_opt_iters : int, optional
        Maximum number of iterations performed in internal optimization.
    """

    def __init__(self, model, threshold=None, priors=None, n_inits=10, max_opt_iters=1000, seed=0):
        super(BolfiPosterior, self).__init__()
        self.threshold = threshold
        self.model = model
        self.random_state = np.random.RandomState(seed)
        self.n_inits = n_inits
        self.max_opt_iters = max_opt_iters

        if priors is None:
            self.priors = [None] * model.input_dim
        else:
            self.priors = priors
            self._prepare_logprior_nets()

        if self.threshold is None:
            minloc, minval = minimize(self.model.predict_mean, self.model.predictive_gradient_mean,
                                      self.model.bounds, self.priors, self.n_inits, self.max_opt_iters,
                                      random_state=self.random_state)
            self.threshold = minval
            logger.info("Using minimum value of discrepancy estimate mean (%.4f) as threshold" % (self.threshold))

    @property
    def ML(self):
        """
        Maximum likelihood (ML) approximation.

        Returns
        -------
        np.array
            Maximum likelihood parameter values.
        """
        x, lh_x = minimize(self._neg_unnormalized_loglikelihood, self._grad_neg_unnormalized_loglikelihood,
                           self.model.bounds, self.priors, self.n_inits, self.max_opt_iters,
                           random_state=self.random_state)
        return x

    @property
    def MAP(self):
        """
        Maximum a posteriori (MAP) approximation.

        Returns
        -------
        np.array
            Maximum a posteriori parameter values.
        """
        x, post_x = minimize(self._neg_unnormalized_logposterior, self._grad_neg_unnormalized_logposterior,
                             self.model.bounds, self.priors, self.n_inits, self.max_opt_iters,
                             random_state=self.random_state)
        return x

    def logpdf(self, x):
        """
        Returns the unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float
        """
        if not self._within_bounds(x):
            return -np.inf
        return self._unnormalized_loglikelihood(x) + self._logprior_density(x)

    def pdf(self, x):
        """
        Returns the unnormalized posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float
        """
        return np.exp(self.logpdf(x))

    def grad_logpdf(self, x):
        """
        Returns the gradient of the unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        np.array
        """
        grad = self._grad_unnormalized_loglikelihood(x) + self._grad_logprior_density(x)
        return grad[0]

    def __getitem__(self, idx):
        return tuple([[v]*len(idx) for v in self.MAP])

    def _unnormalized_loglikelihood(self, x):
        mean, var = self.model.predict(x)
        if mean is None or var is None:
            raise ValueError("Unable to evaluate model at %s" % (x))
        return sp.stats.norm.logcdf(self.threshold, mean, np.sqrt(var))

    def _grad_unnormalized_loglikelihood(self, x):
        mean, var = self.model.predict(x)
        if mean is None or var is None:
            raise ValueError("Unable to evaluate model at %s" % (x))
        std = np.sqrt(var)

        grad_mean, grad_var = self.model.predictive_gradients(x)
        grad_mean = grad_mean[:, :, 0]  # assume 1D output

        factor = -grad_mean * std - (self.threshold - mean) * 0.5 * grad_var / std
        factor = factor / var
        term = (self.threshold - mean) / std
        pdf = sp.stats.norm.pdf(term)
        cdf = sp.stats.norm.cdf(term)

        return factor * pdf / cdf

    def _unnormalized_likelihood(self, x):
        return np.exp(self._unnormalized_loglikelihood(x))

    def _neg_unnormalized_loglikelihood(self, x):
        return -1 * self._unnormalized_loglikelihood(x)

    def _grad_neg_unnormalized_loglikelihood(self, x):
        return -1 * self._grad_unnormalized_loglikelihood(x)

    def _neg_unnormalized_logposterior(self, x):
        return -1 * self.logpdf(x)

    def _grad_neg_unnormalized_logposterior(self, x):
        return -1 * self.grad_logpdf(x)

    def _prepare_logprior_nets(self):
        """Prepares graphs for calculating self._logprior_density and self._grad_logprior_density.
        """
        self._client = elfi.clients.native.Client()  # no need to parallelize here (sampling already parallel)

        # add numerical gradients, if necessary
        for p in self.priors:
            if not hasattr(p.distribution, 'grad_logpdf'):
                p.distribution.grad_logpdf = partial(numerical_grad_logpdf, distribution=p.distribution)

        # augment one model with logpdfs and another with their gradients
        for target in ['logpdf', 'grad_logpdf']:
            model2 = self.priors[0].model.copy()
            outputs = []
            for param in model2.parameters:
                node = model2[param]
                name = '_{}_{}'.format(target, param)
                op = eval('node.distribution.' + target)
                elfi.Operation(op, *([node] + node.parents), name=name, model=model2)
                outputs.append(name)

            # compile and load the new net with data
            context = model2.computation_context.copy()
            context.batch_size = 1  # TODO: need more?
            compiled_net = self._client.compile(model2.source_net, outputs)
            loaded_net = self._client.load_data(compiled_net, context, batch_index=0)

            # remove requests for computation for priors (values assigned later)
            for param in model2.parameters:
                loaded_net.node[param].pop('operation')

            if target == 'logpdf':
                self._logprior_net = loaded_net
            else:
                self._grad_logprior_net = loaded_net

    def _logprior_density(self, x):
        if self.priors[0] is None:
            return 0

        else:  # need to evaluate the graph (up to priors) to account for potential hierarchies

            # load observations for priors
            for i, p in enumerate(self.priors):
                n = p.name
                self._logprior_net.node[n]['output'] = x[i]

            # evaluate graph and return sum of logpdfs
            res = self._client.compute(self._logprior_net)
            return np.array([[sum([v for v in res.values()])]])

    def _within_bounds(self, x):
        x = x.reshape((-1, self.model.input_dim))
        for ii in range(self.model.input_dim):
            if np.any(x[:, ii] < self.model.bounds[ii][0]) or np.any(x[:, ii] > self.model.bounds[ii][1]):
                return False
        return True

    def _grad_logprior_density(self, x):
        if self.priors[0] is None:
            return 0

        else:  # need to evaluate the graph (up to priors) to account for potential hierarchies

            # load observations for priors
            for i, p in enumerate(self.priors):
                n = p.name
                self._grad_logprior_net.node[n]['output'] = x[i]

            # evaluate graph and return components of individual gradients of logpdfs (assumed independent!)
            res = self._client.compute(self._grad_logprior_net)
            res = np.array([[res['_grad_logpdf_' + p.name] for p in self.priors]])
            return res

    def _prior_density(self, x):
        return np.exp(self._logprior_density(x))

    def _neg_logprior_density(self, x):
        return -1 * self._logprior_density(x)

    def plot(self):
        if len(self.model.bounds) == 1:
            mn = self.model.bounds[0][0]
            mx = self.model.bounds[0][1]
            dx = (mx - mn) / 200.0
            x = np.arange(mn, mx, dx)
            pd = np.zeros(len(x))
            for i in range(len(x)):
                pd[i] = self.pdf([x[i]])
            plt.figure()
            plt.plot(x, pd)
            plt.xlim(mn, mx)
            plt.ylim(0.0, max(pd)*1.05)
            plt.show()

        elif len(self.model.bounds) == 2:
            x, y = np.meshgrid(np.linspace(*self.model.bounds[0]), np.linspace(*self.model.bounds[1]))
            z = (np.vectorize(lambda a,b: self.pdf(np.array([a, b]))))(x, y)
            plt.contour(x, y, z)
            plt.show()

        else:
            raise NotImplementedError("Currently unsupported for dim > 2")
