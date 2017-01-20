import numpy as np
from collections import OrderedDict
import inspect
import sys
import io

import elfi.visualization as vis


"""
Implementations related to results and post-processing.
"""


class Result(object):
    """Container for results from ABC methods. Allows intuitive syntax for plotting etc.

    Parameters
    ----------
    samples_list : list of np.arrays
    nodes : list of parameter nodes
    """
    def __init__(self, samples_list, nodes, **kwargs):
        self.samples = OrderedDict()
        for ii, n in enumerate(nodes):
            self.samples[n.name] = samples_list[ii]
        self.n_samples = len(list(self.samples.values())[0])
        self.n_params = len(self.samples)

        # get name of the ABC method
        stack10 = inspect.stack()[1][0]
        self.method = stack10.f_locals["self"].__class__.__name__

        # TODO: remove once interface fixed
        # make sure none of the given kwargs already exist
        for k in kwargs.keys():
            if k in self.__dir__():
                raise KeyError("Conflicting key: ", k)

        self.__dict__.update(kwargs)

    @property
    def samples_list(self):
        """
        Return the samples as a list in the same order as in the OrderedDict samples.

        Returns
        -------
        list of np.arrays
        """
        return list(self.samples.values())

    def __str__(self):
        # create a buffer for capturing the output from summary's print statement
        stdout0 = sys.stdout
        buffer = io.StringIO()
        sys.stdout = buffer
        self.summary()
        sys.stdout = stdout0  # revert to original stdout
        return buffer.getvalue()

    def __repr__(self):
        return self.__str__()

    def summary(self):
        """Print a verbose summary of contained results.
        """
        # TODO: include __str__ of Inference Task, seed?
        desc = "Method: {}\nNumber of posterior samples: {}\n"\
               .format(self.method, self.n_samples)
        if hasattr(self, 'n_sim'):
            desc += "Number of simulations: {}\n".format(self.n_sim)
        if hasattr(self, 'threshold'):
            desc += "Threshold: {:.3g}\n".format(self.threshold)
        desc += self.posterior_means()
        print(desc)

    def posterior_means(self):
        """Return a string representation of posterior means.

        Returns
        -------
        s : string
        """
        s = "Posterior means: "
        s += ', '.join(["{}: {:.3g}".format(k, np.mean(v)) for k,v in self.samples.items()])
        return(s)

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
