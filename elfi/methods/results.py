import io
import logging
import sys
from collections import OrderedDict

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import elfi.visualization.visualization as vis

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
