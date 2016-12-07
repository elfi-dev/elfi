# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as ss
import numpy.random as npr
from functools import partial

from . import core


# TODO: combine with `simulator_wrapper` and perhaps with `summary_wrapper`?
def distribution_wrapper(rvs, input_dict):
    prng = npr.RandomState(0)
    prng.set_state(input_dict['random_state'])
    batch_size = input_dict["n"]
    data = rvs(*input_dict['data'], batch_size=batch_size, random_state=prng)
    return core.to_output_dict(input_dict, data=data, random_state=prng.get_state())


def scipy_rvs_operation(*params, batch_size=1, random_state, distribution_name, size):
    rvs = getattr(getattr(ss, distribution_name), 'rvs')
    size = (batch_size,) + size
    return rvs(*params, size=size, random_state=random_state)


class ElfiDistribution():
    """Must have an attribute or property `name`
    """

    def rvs(self, *params, batch_size=1, random_state):
        raise NotImplementedError

    def pdf(self, x, *params, **kwargs):
        raise NotImplementedError

    def logpdf(self, x, *params, **kwargs):
        raise NotImplementedError

    def cdf(self, x, *params, **kwargs):
        raise NotImplementedError


class ScipyDistribution(ElfiDistribution):

    ALIASES = {'normal': 'norm',
               'exponential': 'expon',
               'unif': 'uniform',
               'bin': 'binom',
               'binomial': 'binom'}

    def __init__(self, distribution, size=(1,)):
        if isinstance(distribution, str):
            distribution = distribution.lower()
            name = self.ALIASES.get(distribution, distribution)
            distribution = getattr(ss, name)
        else:
            name = distribution.name
        self.name = name
        self.scipy_distribution = distribution
        if not isinstance(size, tuple):
            size = (size,)
        self.size = size

    def rvs(self, *params, batch_size=1, random_state):
        rvs = getattr(self.scipy_distribution, 'rvs')
        size = (batch_size,) + self.size
        return rvs(*params, size=size, random_state=random_state)

    def pdf(self, x, *params, **kwargs):
        """Probability density function at x of the given RV.
        """
        if self.is_discrete:
            return self.scipy_distribution.pmf(x, *params, **kwargs)
        else:
            return self.scipy_distribution.pdf(x, *params, **kwargs)

    def logpdf(self, x, *params, **kwargs):
        """Log probability density function at x of the given RV.
        """
        if self.is_discrete:
            return self.scipy_distribution.logpmf(x, *params, **kwargs)
        else:
            return self.scipy_distribution.logpdf(x, *params, **kwargs)

    def cdf(self, x, *params, **kwargs):
        """Cumulative scipy_distribution function of the given RV.
        """
        return self.scipy_distribution.cdf(x, *params, **kwargs)

    @property
    def is_discrete(self):
        return isinstance(self.scipy_distribution, ss.rv_discrete)


class RandomVariable(core.RandomStateMixin, core.Operation):
    """

    Parameters
    ----------
    distribution : string or ElfiDistribution
        string is interpreted as an equivalent scipy distribution
    size : tuple or int
        Size of the RV output

    Examples
    --------
    ScipyRV('tau', scipy.stats.norm, 5, size=(2,3))
    """

    operation_wrapper = distribution_wrapper
    # Convert some common names to scipy equivalents
    ALIASES = {'normal': 'norm',
               'exponential': 'expon',
               'unif': 'uniform',
               'bin': 'binom',
               'binomial': 'binom'}

    def _prepare_operation(self, distribution, size=(1,)):
        if isinstance(distribution, str):
            distribution = distribution.lower()
            name = self.ALIASES.get(distribution, distribution)
            distribution = getattr(ss, name)
        else:
            name = distribution.name
        self.distribution = distribution

        if not isinstance(size, tuple):
            size = (size,)
        return partial(scipy_rvs_operation, distribution_name=name, size=size)

    def __str__(self):
        return super(RandomVariable, self).__str__()[0:-1] + ", '{}')".format(self.distribution.name)


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


class SMC_Distribution():
    """Distribution that samples near previous values of parameters by sampling
    Gaussian distributions centered at previous values.

    Used in SMC ABC as priors for subsequent particle populations.
    """

    def rvs(current_params, weighted_sd, weights, random_state, size=1):
        """Random value source

        Parameters
        ----------
        current_params : 2D np.ndarray
            Expected values for samples. Shape should match weights.
        weighted_sd : float
            Weighted standard deviation to use for the Gaussian.
        weights : 2D np.ndarray
            The probability of selecting each current_params. Shape should match current_params.
        random_state : np.random.RandomState
        size : tuple

        Returns
        -------
        params : np.ndarray
            shape == size
        """
        a = np.arange(current_params.shape[0])
        p = weights[:,0]
        selections = random_state.choice(a=a, size=size, p=p)
        selections = selections[:,0]
        params = current_params[selections]
        noise = ss.norm.rvs(scale=weighted_sd, size=size, random_state=random_state)
        params += noise
        return params

    def pdf(params, current_params, weighted_sd, weights):
        """Probability density function, which is here that of a Gaussian.
        """
        return ss.norm.pdf(params, current_params, weighted_sd)
