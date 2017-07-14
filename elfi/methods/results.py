import io
import logging
import sys
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

import elfi.visualization.visualization as vis

logger = logging.getLogger(__name__)


class ParameterInferenceResult:
    def __init__(self, method_name, outputs, parameter_names, **kwargs):
        """

        Parameters
        ----------
        method_name : string
            Name of inference method.
        outputs : dict
            Dictionary with outputs from the nodes, e.g. samples.
        parameter_names : list
            Names of the parameter nodes
        **kwargs
            Any other information from the inference algorithm, usually from it's state.

        """
        self.method_name = method_name
        self.outputs = outputs.copy()
        self.parameter_names = parameter_names
        self.meta = kwargs


class OptimizationResult(ParameterInferenceResult):
    def __init__(self, x_min, **kwargs):
        """

        Parameters
        ----------
        x_min
            The optimized parameters
        **kwargs
            See `ParameterInferenceResult`

        """
        super(OptimizationResult, self).__init__(**kwargs)
        self.x_min = x_min


class Sample(ParameterInferenceResult):
    """Sampling results from the methods.

    """
    def __init__(self, method_name, outputs, parameter_names, discrepancy_name=None,
                 weights=None, **kwargs):
        """

        Parameters
        ----------
        discrepancy_name : string, optional
            Name of the discrepancy in outputs.
        **kwargs
            Other meta information for the result
        """
        super(Sample, self).__init__(method_name=method_name, outputs=outputs,
                                     parameter_names=parameter_names, **kwargs)

        self.samples = OrderedDict()
        for n in self.parameter_names:
            self.samples[n] = self.outputs[n]

        self.discrepancy_name = discrepancy_name
        self.weights = weights

    def __getattr__(self, item):
        """Allows more convenient access to items under self.meta.
        """
        if item in self.meta.keys():
            return self.meta[item]
        else:
            raise AttributeError("No attribute '{}' in this sample".format(item))

    def __dir__(self):
        """Allows autocompletion for items under self.meta.
        http://stackoverflow.com/questions/13603088/python-dynamic-help-and-autocomplete-generation
        """
        items = dir(type(self)) + list(self.__dict__.keys())
        items.extend(self.meta.keys())
        return items

    @property
    def n_samples(self):
        return len(self.outputs[self.parameter_names[0]])

    @property
    def dim(self):
        return len(self.parameter_names)

    @property
    def discrepancies(self):
        return None if self.discrepancy_name is None else \
            self.outputs[self.discrepancy_name]

    @property
    def samples_array(self):
        """
        Return the samples as an array with columns in the same order as in
        self.parameter_names.

        Returns
        -------
        list of np.arrays
        """
        return np.column_stack(tuple(self.samples.values()))

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
        desc = "Method: {}\nNumber of samples: {}\n"\
               .format(self.method_name, self.n_samples)
        if hasattr(self, 'n_sim'):
            desc += "Number of simulations: {}\n".format(self.n_sim)
        if hasattr(self, 'threshold'):
            desc += "Threshold: {:.3g}\n".format(self.threshold)
        print(desc, end='')
        self.summary_sample_means()

    def summary_sample_means(self):
        """Print a representation of posterior means.
        """
        s = "Sample means: "
        s += ', '.join(["{}: {:.3g}".format(k, v) for k,v in self.sample_means.items()])
        print(s)

    @property
    def sample_means(self):
        return OrderedDict([(k, np.average(v, axis=0, weights=self.weights)) for \
                            k,v in self.samples.items()])

    @property
    def sample_means_array(self):
        return np.array(list(self.sample_means.values()))

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


class SmcSample(Sample):
    """Container for results from SMC-ABC.
    """
    def __init__(self, method_name, outputs, parameter_names, populations, *args,
                 **kwargs):
        """

        Parameters
        ----------
        method_name : str
        outputs : dict
        parameter_names : list
        populations : list[Sample]
            List of Sample objects
        args
        kwargs
        """
        super(SmcSample, self).__init__(method_name=method_name, outputs=outputs,
                                        parameter_names=parameter_names, *args, **kwargs)
        self.populations = populations

        if self.weights is None:
            raise ValueError("No weights provided for the sample")

    @property
    def n_populations(self):
        return len(self.populations)

    @property
    def sample_means_all_populations(self):
        """Return a list of sample means for all populations
        """
        means = []
        for i in range(self.n_populations):
            means.append(self.populations[i].sample_means)
        return means

    def summary_sample_means_all_populations(self):
        out = ''
        for i, means in enumerate(self.sample_means_all_populations):
            out += "Sample means for population {}: ".format(i)
            out += ', '.join(["{}: {:.3g}".format(k, v) for k, v in means.items()])
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


class BolfiSample(Sample):
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

        super(BolfiSample, self).__init__(method_name=method_name, outputs=outputs,
                                          parameter_names=parameter_names,
                                          chains=chains, n_chains=n_chains, warmup=warmup,
                                          **kwargs)

    def plot_traces(self, selector=None, axes=None, **kwargs):
        return vis.plot_traces(self, selector, axes, **kwargs)
