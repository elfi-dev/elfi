import io
import logging
import sys
from collections import OrderedDict

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import elfi.visualization.visualization as vis
from elfi.methods.bo.utils import stochastic_optimization

logger = logging.getLogger(__name__)


"""
Implementations related to results and post-processing.
"""


class Result(object):
    """Container for results from ABC methods. Allows intuitive syntax for plotting etc.

    Parameters
    ----------
    method_name : string
        Name of inference method.
    outputs : dict
        Dictionary with values as np.arrays. May contain more keys than just the names of priors.
    parameter_names : list : list of strings
        List of names in the outputs dict that refer to model parameters.
    discrepancy_name : string, optional
        Name of the discrepancy in outputs.
    """
    # TODO: infer these from state?
    def __init__(self, method_name, outputs, parameter_names, discrepancy_name=None, **kwargs):
        self.method_name = method_name
        self.outputs = outputs.copy()
        self.samples = OrderedDict()

        for n in parameter_names:
            self.samples[n] = outputs[n]
        if discrepancy_name is not None:
            self.discrepancy = outputs[discrepancy_name]

        self.n_samples = len(outputs[parameter_names[0]])
        self.n_params = len(parameter_names)

        # store arbitrary keyword arguments here
        self.meta = kwargs

    def __getattr__(self, item):
        """Allows more convenient access to items under self.meta.
        """
        if item in self.__dict__:
            return self.item
        elif item in self.meta.keys():
            return self.meta[item]
        else:
            raise AttributeError("No attribute '{}' in this Result".format(item))

    def __dir__(self):
        """Allows autocompletion for items under self.meta.
        http://stackoverflow.com/questions/13603088/python-dynamic-help-and-autocomplete-generation
        """
        items = dir(type(self)) + list(self.__dict__.keys())
        items.extend(self.meta.keys())
        return items

    @property
    def samples_list(self):
        """
        Return the samples as a list in the same order as in the OrderedDict samples.

        Returns
        -------
        list of np.arrays
        """
        return list(self.samples.values())

    @property
    def names_list(self):
        """
        Return the parameter names as a list in the same order as in the OrderedDict samples.

        Returns
        -------
        list of strings
        """
        return list(self.samples.keys())

    def __str__(self):
        # create a buffer for capturing the output from summary's print statement
        stdout0 = sys.stdout
        buffer = io.StringIO()
        sys.stdout = buffer
        self.summary
        sys.stdout = stdout0  # revert to original stdout
        return buffer.getvalue()

    def __repr__(self):
        return self.__str__()

    @property
    def summary(self):
        """Print a verbose summary of contained results.
        """
        # TODO: include __str__ of Inference Task, seed?
        desc = "Method: {}\nNumber of posterior samples: {}\n"\
               .format(self.method_name, self.n_samples)
        if hasattr(self, 'n_sim'):
            desc += "Number of simulations: {}\n".format(self.n_sim)
        if hasattr(self, 'threshold'):
            desc += "Threshold: {:.3g}\n".format(self.threshold)
        print(desc, end='')
        self.posterior_means

    @property
    def posterior_means(self):
        """Print a representation of posterior means.
        """
        s = "Posterior means: "
        s += ', '.join(["{}: {:.3g}".format(k, np.mean(v)) for k,v in self.samples.items()])
        print(s)

    def plot_marginals(self, selector=None, bins=20, axes=None, **kwargs):
        """Plot marginal distributions for parameters.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional

        Returns
        -------
        axes : np.array of plt.Axes
        """
        return vis.plot_marginals(self.samples, selector, bins, axes, **kwargs)

    def plot_pairs(self, selector=None, bins=20, axes=None, **kwargs):
        """Plot pairwise relationships as a matrix with marginals on the diagonal.

        The y-axis of marginal histograms are scaled.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional

        Returns
        -------
        axes : np.array of plt.Axes
        """
        return vis.plot_pairs(self.samples, selector, bins, axes, **kwargs)


class ResultSMC(Result):
    """Container for results from SMC-ABC.
    """
    def __init__(self, *args, **kwargs):
        super(ResultSMC, self).__init__(*args, **kwargs)
        self.n_populations = len(self.populations)

    @property
    def posterior_means(self):
        """Print a representation of posterior means.
        """
        s = self.populations[-1].samples_list
        w = self.populations[-1].weights
        n = self.names_list
        out = ''
        out += "Posterior means for final population: "
        out += ', '.join(["{}: {:.3g}".format(n[jj], np.average(s[jj], weights=w, axis=0))
                          for jj in range(self.n_params)])
        print(out)

    @property
    def posterior_means_all_populations(self):
        """Print a representation of posterior means for all populations.

        Returns
        -------
        out : string
        """
        samples = [pop.samples_list for pop in self.populations]
        weights = [pop.weights for pop in self.populations]
        n = self.names_list
        out = ''
        for ii in range(self.n_populations):
            s = samples[ii]
            w = weights[ii]
            out += "Posterior means for population {}: ".format(ii)
            out += ', '.join(["{}: {:.3g}".format(n[jj], np.average(s[jj], weights=w, axis=0))
                              for jj in range(self.n_params)])
            out += '\n'
        print(out)

    def plot_marginals_all_populations(self, selector=None, bins=20, axes=None, **kwargs):
        """Plot marginal distributions for parameters for all populations.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional
        """
        samples = [pop.samples_list for pop in self.populations]
        fontsize = kwargs.pop('fontsize', 13)
        for ii in range(self.n_populations):
            s = OrderedDict()
            for jj, n in enumerate(self.names_list):
                s[n] = samples[ii][jj]
            ax = vis.plot_marginals(s, selector, bins, axes, **kwargs)
            plt.suptitle("Population {}".format(ii), fontsize=fontsize)

    def plot_pairs_all_populations(self, selector=None, bins=20, axes=None, **kwargs):
        """Plot pairwise relationships as a matrix with marginals on the diagonal for all populations.

        The y-axis of marginal histograms are scaled.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        bins : int, optional
            Number of bins in histograms.
        axes : one or an iterable of plt.Axes, optional
        """
        samples = self.samples_history + [self.samples_list]
        fontsize = kwargs.pop('fontsize', 13)
        for ii in range(self.n_populations):
            s = OrderedDict()
            for jj, n in enumerate(self.names_list):
                s[n] = samples[ii][jj]
            ax = vis.plot_pairs(s, selector, bins, axes, **kwargs)
            plt.suptitle("Population {}".format(ii), fontsize=fontsize)


class ResultBOLFI(Result):
    """Container for results from BOLFI.

    Parameters
    ----------
    method_name : string
        Name of inference method.
    chains : np.array
        Chains from sampling. Shape should be (n_chains, n_samples, n_parameters) with warmup included.
    parameter_names : list : list of strings
        List of names in the outputs dict that refer to model parameters.
    warmup : int
        Number of warmup iterations in chains.
    """
    def __init__(self, method_name, chains, parameter_names, warmup, **kwargs):
        chains = chains.copy()
        shape = chains.shape
        n_chains = shape[0]
        warmed_up = chains[:, warmup:, :]
        concatenated = warmed_up.reshape((-1,) + shape[2:])
        outputs = dict(zip(parameter_names, concatenated.T))

        super(ResultBOLFI, self).__init__(method_name=method_name, outputs=outputs, parameter_names=parameter_names,
                                           chains=chains, n_chains=n_chains, warmup=warmup, **kwargs)

    def plot_traces(self, selector=None, axes=None, **kwargs):
        return vis.plot_traces(self, selector, axes, **kwargs)


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
    max_opt_iters : int, optional
        Maximum number of iterations performed in internal optimization.
    """

    def __init__(self, model, threshold=None, priors=None, max_opt_iters=10000):
        super(BolfiPosterior, self).__init__()
        self.threshold = threshold
        self.model = model
        if self.threshold is None:
            minloc, minval = stochastic_optimization(self.model.predict_mean, self.model.bounds, max_opt_iters)
            self.threshold = minval
            logger.info("Using minimum value of discrepancy estimate mean (%.4f) as threshold" % (self.threshold))
        self.priors = priors or [None] * model.input_dim
        self.max_opt_iters = max_opt_iters

    @property
    def ML(self):
        """
        Maximum likelihood (ML) approximation.

        Returns
        -------
        tuple
            Maximum likelihood parameter values and the corresponding value of neg_unnormalized_loglikelihood.
        """
        x, lh_x = stochastic_optimization(self._neg_unnormalized_loglikelihood,
                                          self.model.bounds, self.max_opt_iters)
        return x, lh_x

    @property
    def MAP(self):
        """
        Maximum a posteriori (MAP) approximation.

        Returns
        -------
        tuple
            Maximum a posteriori parameter values and the corresponding value of neg_unnormalized_logposterior.
        """
        x, post_x = stochastic_optimization(self._neg_unnormalized_logposterior,
                                            self.model.bounds, self.max_opt_iters)
        return x, post_x

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

    def _neg_unnormalized_logposterior(self, x):
        return -1 * self.logpdf(x)

    def _logprior_density(self, x):
        logprior_density = 0.0
        for xv, prior in zip(x, self.priors):
            if prior is not None:
                logprior_density += prior.logpdf(xv)
        return logprior_density

    def _within_bounds(self, x):
        x = x.reshape((-1, self.model.input_dim))
        for ii in range(self.model.input_dim):
            if np.any(x[:, ii] < self.model.bounds[ii][0]) or np.any(x[:, ii] > self.model.bounds[ii][1]):
                return False
        return True

    def _grad_logprior_density(self, x):
        grad_logprior_density = np.zeros(x.shape)
        for ii, prior in enumerate(self.priors):
            if prior is not None:
                grad_logprior_density[ii] = prior.grad_logpdf(x[ii])
        return grad_logprior_density

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
            raise NotImplementedError("Currently not supported for dim > 2")