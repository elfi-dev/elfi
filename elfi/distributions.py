# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
import numpy.random as npr
from functools import partial

from . import core
from . import utils


# TODO: combine with `simulator_wrapper` and perhaps with `summary_wrapper`?
def random_transform(input_dict, operation):
    random_state = npr.RandomState(0)
    random_state.set_state(input_dict['random_state'])
    batch_size = input_dict["n"]
    data = operation(*input_dict['data'], batch_size=batch_size, random_state=random_state)
    return core.to_output_dict(input_dict, data=data, random_state=random_state.get_state())


class Distribution:
    """Must have an attribute or property `name`
    """

    def __init__(self, name=None):
        self.name = name

    def rvs(self, *params, size=(1,), random_state):
        raise NotImplementedError

    def pdf(self, x, *params, **kwargs):
        raise NotImplementedError

    def logpdf(self, x, *params, **kwargs):
        raise NotImplementedError


# TODO: this might be needed for rv_discrete instances in the future?
class ScipyDistribution(Distribution):

    # Convert some common names to scipy equivalents
    ALIASES = {'normal': 'norm',
               'exponential': 'expon',
               'unif': 'uniform',
               'bin': 'binom',
               'binomial': 'binom'}

    def __init__(self, distribution):
        if isinstance(distribution, str):
            distribution = self.__class__.from_str(distribution)
        elif not isinstance(distribution, (ss.rv_continuous, ss.rv_discrete)):
            raise ValueError("Unknown distribution type {}".format(distribution))

        self.ss_distribution = distribution

        name = distribution.name
        super(ScipyDistribution, self).__init__(name=name)

    def rvs(self, *params, size=1, random_state=None):
        return self.ss_distribution.rvs(*params, size=size, random_state=random_state)

    def pdf(self, x, *params, **kwargs):
        """Probability density function at x of the given RV.
        """
        if self.is_discrete:
            return self.ss_distribution.pmf(x, *params, **kwargs)
        else:
            return self.ss_distribution.pdf(x, *params, **kwargs)

    def logpdf(self, x, *params, **kwargs):
        """Log probability density function at x of the given RV.
        """
        if self.is_discrete:
            return self.ss_distribution.logpmf(x, *params, **kwargs)
        else:
            return self.ss_distribution.logpdf(x, *params, **kwargs)

    def cdf(self, x, *params, **kwargs):
        """Cumulative scipy_distribution function of the given RV.
        """
        return self.ss_distribution.cdf(x, *params, **kwargs)

    @property
    def is_discrete(self):
        return isinstance(self.ss_distribution, ss.rv_discrete)

    @classmethod
    def from_str(cls, name):
        name = name.lower()
        name = cls.ALIASES.get(name, name)
        return getattr(ss, name)


def rvs_operation(*params, batch_size=1, random_state, distribution, size=(1,)):
    size = (batch_size,) + size
    return distribution.rvs(*params, size=size, random_state=random_state)


class RandomVariable(core.RandomStateMixin, core.Operation):
    """

    Parameters
    ----------
    distribution : string or Distribution
        string is interpreted as an equivalent scipy distribution
    size : tuple or int
        Size of the RV output

    Examples
    --------
    RandomVariable('tau', scipy.stats.norm, 5, size=(2,3))
    """

    operation_transform = random_transform

    def _prepare_operation(self, distribution, size=(1,), **kwargs):
        if isinstance(distribution, str):
            distribution = ScipyDistribution.from_str(distribution)
        if not hasattr(distribution, 'rvs'):
            raise ValueError("Distribution {} must implement rvs method".format(distribution))

        if not isinstance(size, tuple):
            size = (size,)

        self.distribution = distribution
        return partial(rvs_operation, distribution=distribution, size=size)

    def __str__(self):
        return super(RandomVariable, self).__str__()[0:-1] + \
               ", '{}')".format(self.distribution.name)


class Prior(RandomVariable):
    def __init__(self, name, distribution="uniform", *args, **kwargs):
        super(Prior, self).__init__(name, distribution, *args, **kwargs)


class Model(core.ObservedMixin, RandomVariable):
    def __init__(self, *args, observed=None, size=None, **kwargs):
        if observed is None:
            raise ValueError('Observed cannot be None')
        if size is None:
            size = observed.shape
        super(Model, self).__init__(*args, observed=observed, size=size, **kwargs)


class SMCProposal():
    """Distribution that samples near previous values of parameters by sampling
    Gaussian distributions centered at previous values.

    Used in SMC ABC as priors for subsequent particle populations.
    """

    def __init__(self, samples=None, weights=None):
        """

        Parameters
        ----------
        samples : 2-D array-like, optional
            Observations in rows
        weights : 1-D array-like or float, optional
        """

        self._samples = None
        self.weights = None
        self.set_population(samples, weights)

    def set_population(self, samples, weights):
        self._samples = utils.atleast_2d(samples)
        self.weights = weights

    def resample(self, size=1, random_state=None):
        if isinstance(size, tuple):
            if len(size) > 1:
                raise ValueError('Size cannot be multidimensional')
            size = size[0]

        if random_state is None:
            random_state = np.random

        inds = random_state.choice(len(self._samples), size=size, p=self.p_weights)
        return self._samples[inds]

    def rvs(self, size=1, random_state=None):
        """Random value source

        Parameters
        ----------
        size : int or tuple
        random_state : np.random.RandomState

        Returns
        -------
        np.ndarray

        """

        samples = self.resample(size=size, random_state=random_state).astype(core.DEFAULT_DATATYPE)
        samples += ss.multivariate_normal.rvs(cov=self._cov, random_state=random_state)
        return samples

    def pdf(self, x):
        x = utils.atleast_2d(x)
        vals = np.zeros(len(x))
        d = ss.multivariate_normal(mean=[0]*self._samples.shape[1], cov=self._cov)
        for i in range(len(x)):
            xi = x[i,:] - self._samples
            vals[i] = np.sum(self.p_weights * d.pdf(xi))
        return vals

    @property
    def _cov(self):
        return 2*np.cov(self._samples, rowvar=False)

    @property
    def samples(self):
        return self._samples

    @property
    def p_weights(self):
        p = self.weights / np.sum(self.weights)
        if p.ndim == 0:
            l = len(self._samples)
            p = np.ones(l) / l
        return p
