import numpy as np
from collections import OrderedDict

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
        self.n_samples = len(samples)

        # TODO: needed?
        if any(map(lambda k: k in self.__dir__(), kwargs.keys())):
            raise KeyError("Conflicting key")

        self.__dict__.update(kwargs)

    def __str__(self):
        # TODO: change to a summary including __str__ of Inference Task?
        return self.posterior_means(stdout=False)

    def posterior_means(self, stdout=True):
        """Evaluate a string representation of posterior means.
        """
        s = "Posterior means: "
        for k, v in self.samples.items():
            s += "{}: {:.3g}, ".format(k, np.mean(v))
        s = s[:-1]
        if stdout:
            print(s)
        else:
            return s

    def plot_marginals(self, selector=None, axes=None, **kwargs):
        """Plot marginal distributions for parameters.

        Parameters
        ----------
        selector : iterable of ints or strings, optional
            Indices or keys to use from samples. Default to all.
        axes : one or an iterable of plt.Axes, optional

        Returns
        -------
        axes : np.array of plt.Axes
        """
        if selector is None:
            selected = self.samples
        else:
            selected = OrderedDict()
            for ii, k in enumerate(self.samples):
                if ii in selector or k in selector:
                    selected[k] = self.samples[k]

        return vis.plot_histogram(selected, axes, **kwargs)
