# -*- coding: utf-8 -*-
import scipy.stats as ss
import numpy.random as npr
from functools import partial

from . import core


def npr_op(distribution, size, input):
    prng = npr.RandomState(0)
    prng.set_state(input['random_state'])
    distribution = getattr(prng, distribution)
    size = (input['n'], *size)
    data = distribution(*input['data'], size=size)
    return core.to_output(input, data=data, random_state=prng.get_state())


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
        super(NumpyRV, self).__init__(name, op, *params)


class Prior(NumpyRV):
   pass


class Model(core.ObservedMixin, NumpyRV):
    def __init__(self, *args, observed=None, size=None, **kwargs):
        if observed is None:
            raise ValueError('Observed cannot be None')
        if size is None:
            size = observed.shape
        super(Model, self).__init__(*args, observed=observed, size=size, **kwargs)
