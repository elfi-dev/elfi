# -*- coding: utf-8 -*-
import scipy.stats as ss
import numpy.random as npr
from functools import partial

from . import core


def npr_op(distribution, size, input_dict):
    prng = npr.RandomState(0)
    prng.set_state(input_dict['random_state'])
    distribution = getattr(prng, distribution)
    size = (input_dict['n'],)+size
    data = distribution(*input_dict['data'], size=size)
    return core.to_output(input_dict, data=data, random_state=prng.get_state())


class NumpyRV(core.RandomStateMixin, core.Operation):
    """

    Examples
    --------
    NumpyRV('tau', 'normal', 5, size=(2,3))

    """
    def __init__(self, name, distribution, *params, size=(1,)):
        if not isinstance(size, tuple):
            size = (size,)
        op = partial(npr_op, distribution, size)
        super(NumpyRV, self).__init__(name, op, *params, random_state=prng.get_state())


def spr_op(distribution, size, args):
    prng = npr.RandomState(0)
    prng.set_state(args['random_state'])
    size = (args['n'],)+tuple(size)
    data = distribution.rvs(*args['data'], size=size, random_state=prng)
    return core.to_output(args, data=data, random_state=prng.get_state())


class ScipyRV_cont(core.RandomStateMixin, core.Operation):
    """
    Allows any distribution inheriting scipy.stats.rv_continuous

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
        super(ScipyRV_cont, self).__init__(name, op, *params)

    def pdf(self, x):
        """
        Probability density function at x of the given RV.
        """
        return self.distribution.pdf(x, *self.params)

    def logpdf(self, x):
        """
        Log probability density function at x of the given RV.
        """
        return self.distribution.logpdf(x, *self.params)

    def cdf(self, x):
        """
        Cumulative distribution function of the given RV.
        """
        return self.distribution.cdf(x, *self.params)


# class Prior(NumpyRV):
class Prior(ScipyRV_cont):
    pass


class Model(core.ObservedMixin, NumpyRV):
    def __init__(self, *args, observed=None, size=None, **kwargs):
        if observed is None:
            raise ValueError('Observed cannot be None')
        if size is None:
            size = observed.shape
        super(Model, self).__init__(*args, observed=observed, size=size, **kwargs)
