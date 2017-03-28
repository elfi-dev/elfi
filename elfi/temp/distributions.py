# -*- coding: utf-8 -*-
import logging
from functools import partial

import numpy as np
import numpy.random as npr
import scipy.stats as ss

from elfi import core
from elfi import utils
from elfi.utils import weighted_cov, normalize_weights

logger = logging.getLogger(__name__)


# TODO: combine with `simulator_transform` and perhaps with `summary_transform`?
def random_transform(input_dict, operation):
    """Provides operation a RandomState object for generating random quantities.

    Parameters
    ----------
    input_dict: dict
        ELFI input_dict for transformations
    operation: callable(*parent_data, batch_size, random_state)
        parent_data1, parent_data2, ... : np.ndarray
        batch_size : int
            number of simulations to perform
        random_state : RandomState

    Notes
    -----
    It is crucial to use the provided RandomState object for generating the random
    quantities when running the simulator. This ensures that results are reproducible and
    inference will be valid.

    If the simulator is implemented in another language, one should extract the state
    of the random_state and use it in generating the random numbers.

    """
    random_state = npr.RandomState(0)
    random_state.set_state(input_dict['random_state'])
    batch_size = input_dict["n"]
    data = operation(*input_dict['data'], batch_size=batch_size, random_state=random_state)
    return core.to_output_dict(input_dict, data=data, random_state=random_state.get_state())


class Distribution:
    """Abstract class for an ELFI compatible random distribution.

    Note that the class signature is a subset of that of `scipy.rv_continuous`
    """

    def __init__(self, name=None):
        """

        Parameters
        ----------
        name : name of the distribution
        """
        self.name = name or self.__class__.__name__

    def rvs(self, *params, size=(1,), random_state):
        """Random variates

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

    def pdf(self, x, *params, **kwargs):
        """Probability density function at x

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

    def logpdf(self, x, *params, **kwargs):
        """Log of the probability density function at x.

        Parameters
        ----------
        x : array_like
            points where to evaluate the pdf
        param1, param2, ... : array_like
            parameters of the model
        kwargs

        Returns
        -------
        pdf : ndarray
           Log of the probability density function evaluated at x
        """
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


def rvs_operation(*params, distribution, random_state, batch_size=1, size=(1,)):
    """

    Parameters
    ----------
    params
        Parameters for the distribution
    distribution : scipy-like distribution object
    random_state : RandomState object
    batch_size : int
    size : tuple
        Size of a single datum from the distribution.

    Returns
    -------
    random variates from the distribution

    Notes
    -----
    Used internally by the RandomVariable to wrap distributions for the framework.

    """
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

    def _init_transform(self, distribution, size=(1,), **kwargs):
        if not isinstance(size, tuple):
            size = (size,)

        if isinstance(distribution, str):
            distribution = ScipyDistribution.from_str(distribution)
        if not hasattr(distribution, 'rvs'):
            raise ValueError("Distribution {} "
                             "must implement a rvs method".format(distribution))

        self.distribution = distribution
        operation = partial(rvs_operation, distribution=distribution, size=size)
        return super(RandomVariable, self)._init_transform(operation, **kwargs)

    def __str__(self):
        d = self.distribution

        if hasattr(d, 'name'):
            name = d.name
        elif isinstance(d, type):
            name = self.distribution.__name__
        else:
            name = self.distribution.__class__.__name__

        return super(RandomVariable, self).__str__()[0:-1] + ", '{}')".format(name)


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

    def __init__(self, samples=None, weights=1):
        """

        Parameters
        ----------
        samples : 2-D array-like, optional
            Observations in rows
        weights : 1-D array-like or numeric, optional
        """
        self._samples = None
        self._weights = None
        self._wcov = None
        self.set_population(samples, weights)

    def set_population(self, samples, weights=1):
        self._samples = utils.atleast_2d(samples).astype(core.DEFAULT_DATATYPE)

        weights = normalize_weights(weights)
        if weights.ndim == 0 or len(weights) == 1:
            # Scalar case
            l = len(self._samples)
            weights = np.ones(l) / l
        elif 1 < len(weights) != len(self._samples):
            raise ValueError("Weights do not match to the number of samples")

        self._weights = weights
        self._wcov = weighted_cov(self._samples, self._weights)

    def resample(self, size=(1,), random_state=None):
        size = self._size_to_int(size)

        if random_state is None:
            random_state = np.random

        inds = random_state.choice(len(self._samples), size=size, p=self._weights)
        return self._samples[inds]

    def rvs(self, size=(1,), random_state=None):
        """Random value source

        Parameters
        ----------
        size : int or tuple
        random_state : np.random.RandomState

        Returns
        -------
        samples : np.ndarray

        """
        size = self._size_to_int(size)

        samples = self.resample(size=size,
                                random_state=random_state).astype(core.DEFAULT_DATATYPE)

        noise = ss.multivariate_normal.rvs(cov=2*self._wcov,
                                           random_state=random_state,
                                           size=size)
        if size == 1:
            noise = np.atleast_2d(noise)
        samples += utils.atleast_2d(noise)

        return samples

    def pdf(self, x):
        x = utils.atleast_2d(x)
        vals = np.zeros(len(x))
        d = ss.multivariate_normal(mean=[0]*self._samples.shape[1],
                                   cov=2*self._wcov)
        for i in range(len(x)):
            xi = x[i,:] - self._samples
            vals[i] = np.sum(self._weights * d.pdf(xi))
        return vals

    @property
    def size(self):
        return self._samples[0].shape

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def weights(self):
        return self._weights.copy()

    def _size_to_int(self, size):
        if isinstance(size, tuple):
            if self.size != size[1:]:
                raise ValueError('Requested size {} does not match '
                                 'with the sample size {}'.format(size[1:], self.size))
            size = size[0]
        return size
