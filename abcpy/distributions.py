# -*- coding: utf-8 -*-
import scipy.stats as ss
import numpy.random as npr
from functools import partial

from . import core


def spr_op(distribution, size, input_dict):
    prng = npr.RandomState(0)
    prng.set_state(input_dict['random_state'])
    size = (input_dict['n'],)+tuple(size)
    data = distribution.rvs(*input_dict['data'], size=size, random_state=prng)
    return core.to_output(input_dict, data=data, random_state=prng.get_state())


class ScipyRV(core.RandomStateMixin, core.Operation):
    """
    Allows any distribution inheriting scipy.stats.rv_continuous or
    scipy.stats.rv_discrete. In the latter case methods pdf and logpdf
    are mapped to pmf and logpmf.

    Examples
    --------
    ScipyRV_cont('tau', scipy.stats.norm, 5, size=(2,3))
    """
    def __init__(self, name, distribution, *params, size=(1,)):
        self.distribution = distribution
        self.params = params
        if not isinstance(size, tuple):
            size = (size,)
        op = partial(spr_op, distribution, size)
        super(ScipyRV, self).__init__(name, op, *params)

    def pdf(self, x):
        """
        Probability density function at x of the given RV.
        """
        if isinstance(self, ss.rv_discrete):
            return self.distribution.pmf(x, *self.params)
        else:
            return self.distribution.pdf(x, *self.params)

    def logpdf(self, x):
        """
        Log probability density function at x of the given RV.
        """
        if isinstance(self, ss.rv_discrete):
            return self.distribution.logpmf(x, *self.params)
        else:
            return self.distribution.logpdf(x, *self.params)

    def cdf(self, x):
        """
        Cumulative distribution function of the given RV.
        """
        return self.distribution.cdf(x, *self.params)


class Prior(ScipyRV):
    pass


class Model(core.ObservedMixin, ScipyRV):
    def __init__(self, *args, observed=None, size=None, **kwargs):
        if observed is None:
            raise ValueError('Observed cannot be None')
        if size is None:
            size = observed.shape
        super(Model, self).__init__(*args, observed=observed, size=size, **kwargs)
