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
    return core.to_output_dict(input_dict, data=data, random_state=prng.get_state())


class ScipyRV(core.RandomStateMixin, core.Operation):
    """
    Allows any distribution inheriting scipy.stats.rv_continuous or
    scipy.stats.rv_discrete. In the latter case methods pdf and logpdf
    are mapped to pmf and logpmf.

    Examples
    --------
    ScipyRV('tau', scipy.stats.norm, 5, size=(2,3))
    """

    # Convert some common names to scipy equivalents
    ALIASES = {'normal': 'norm'}

    def __init__(self, name, distribution, *params, size=(1,), **kwargs):
        if isinstance(distribution, str):
            distribution = distribution.lower()
            distribution = getattr(ss, self.ALIASES.get(distribution, distribution))
        self.distribution = distribution
        if not isinstance(size, tuple):
            size = (size,)
        op = partial(spr_op, distribution, size)
        super(ScipyRV, self).__init__(name, op, *params, **kwargs)

    @property
    def is_discrete(self):
        return isinstance(self.distribution, ss.rv_discrete)

    def pdf(self, x, *params):
        """
        Probability density function at x of the given RV.
        """
        params = self._get_params(*params)
        if self.is_discrete:
            return self.distribution.pmf(x, *params)
        else:
            return self.distribution.pdf(x, *params)

    def logpdf(self, x, *params):
        """
        Log probability density function at x of the given RV.
        """
        params = self._get_params(*params)
        if self.is_discrete:
            return self.distribution.logpmf(x, *params)
        else:
            return self.distribution.logpdf(x, *params)

    def cdf(self, x, *params):
        """
        Cumulative distribution function of the given RV.
        """
        params = self._get_params(*params)
        return self.distribution.cdf(x, *params)

    def _get_params(self, *arg_params):
        """
        Parses constant params from the parents and adds arg_params to non constant params
        """
        arg_params = list(arg_params)
        params = []
        for i, p in enumerate(self.parents):
            if isinstance(p, core.Constant):
                params.append(p.value)
            elif len(arg_params) > 0:
                params.append(arg_params.pop(0))
            else:
                raise IndexError('Not enough parameters provided')
        if len(arg_params) > 0:
            raise ValueError('Too many params provided')
        return params


class Prior(ScipyRV):
    pass


class Model(core.ObservedMixin, ScipyRV):
    def __init__(self, *args, observed=None, size=None, **kwargs):
        if observed is None:
            raise ValueError('Observed cannot be None')
        if size is None:
            size = observed.shape
        super(Model, self).__init__(*args, observed=observed, size=size, **kwargs)
