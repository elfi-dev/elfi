import numpy as np
from collections import OrderedDict
import inspect

import elfi.visualization as vis


"""
Implementations for post-processing.
"""


class Result(object):
    """Container for results from ABC methods. Allows intuitive syntax for plotting etc.

    Parameters
    ----------
    samples : OrderedDict of name: np.arrays
    """
    def __init__(self, samples, **kwargs):
        self.samples = samples
        self.n_samples = len(list(self.samples.values())[0])
        self.n_params = len(samples)

        # get name of the ABC method
        stack10 = inspect.stack()[1][0]
        self.method = stack10.f_locals["self"].__class__.__name__

        # TODO: needed?
        if any(map(lambda k: k in self.__dir__(), kwargs.keys())):
            raise KeyError("Conflicting key")

        self.__dict__.update(kwargs)

    def __str__(self):
        return self.summarize(stdout=False)

    def __repr__(self):
        return self.__str__()

    def summarize(self, stdout=True):
        # TODO: include __str__ of Inference Task, seed?
        desc = "Method: {}\nNumber of posterior samples: {}\n"\
               .format(self.method, self.n_samples)
        if hasattr(self, 'n_sim'):
            desc += "Number of simulations: {}\n".format(self.n_sim)
        if hasattr(self, 'threshold'):
            desc += "Threshold: {:.3g}\n".format(self.threshold)
        desc += self.posterior_means(stdout=False)
        if stdout:
            print(desc)
        else:
            return desc

    def posterior_means(self, stdout=True):
        """A string representation of posterior means.

        Parameters
        ----------
        stdout : bool
            If True, print string, else return it.
        """
        s = "Posterior means:"
        for k, v in self.samples.items():
            s += " {}: {:.3g},".format(k, np.mean(v))
        s = s[:-1]
        if stdout:
            print(s)
        else:
            return s

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
