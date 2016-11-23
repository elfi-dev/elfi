# -*- coding: utf-8 -*-
import numpy as np
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
    Allows any `distribution` inheriting/similar to scipy.stats.rv_continuous or
    scipy.stats.rv_discrete. In the latter case methods pdf and logpdf
    are mapped to pmf and logpmf.

    Parameters
    ----------
    distribution : a static class
        A static class that implements a method rvs and optionally: pdf, logpdf, cdf
        similar to inheriting scipy.stats.rv_continuous.
    is_discrete : boolean, optional
        Whether the distribution is discrete (default: False)

    Examples
    --------
    ScipyRV('tau', scipy.stats.norm, 5, size=(2,3))
    """

    # Convert some common names to scipy equivalents
    ALIASES = {'normal': 'norm',
               'exponential': 'expon',
               'unif': 'uniform',
               'bin': 'binom',
               'binomial': 'binom'}

    def __init__(self, name, distribution, *params, size=(1,), is_discrete=False, **kwargs):
        if isinstance(distribution, str):
            distribution = distribution.lower()
            distribution = getattr(ss, self.ALIASES.get(distribution, distribution))
        self.distribution = distribution
        if not isinstance(size, tuple):
            size = (size,)
        self.is_discrete = is_discrete
        op = partial(spr_op, distribution, size)

        super(ScipyRV, self).__init__(name, op, *params, **kwargs)

    @property
    def is_conditional(self):
        """Tell if the Prior is conditional on some other random node.
        This may change if parents change.
        """
        for p in self.parents:
            if isinstance(p, core.RandomStateMixin):
                return True
        return False

    def pdf(self, x, *params, **kwargs):
        """Probability density function at x of the given RV.
        """
        params = self._get_params(params, x.shape[0])
        kwargs = self._get_kwargs(kwargs)

        if self.is_discrete:
            return self.distribution.pmf(x, *params, **kwargs)
        else:
            return self.distribution.pdf(x, *params, **kwargs)

    def logpdf(self, x, *params, **kwargs):
        """Log probability density function at x of the given RV.
        """
        params = self._get_params(params, x.shape[0])
        kwargs = self._get_kwargs(kwargs)

        if self.is_discrete:
            return self.distribution.logpmf(x, *params, **kwargs)
        else:
            return self.distribution.logpdf(x, *params, **kwargs)

    def cdf(self, x, *params, **kwargs):
        """Cumulative distribution function of the given RV.
        """
        params = self._get_params(params, x.shape[0])
        kwargs = self._get_kwargs(kwargs)

        return self.distribution.cdf(x, *params, **kwargs)

    def _get_params(self, arg_params, n):
        """Parses constant params from constants parents, acquires samples from random parents.
        Appends other arguments.
        """
        params = []
        for i, p in enumerate(self.parents):
            if isinstance(p, core.Constant):
                params.append(p.value)
            elif isinstance(p, core.RandomStateMixin):
                params.append(p.acquire(n).compute())
            elif len(arg_params) > 0:
                params.append(arg_params.pop(0))
            else:
                raise IndexError('Not enough parameters provided')
        if len(arg_params) > 0:
            raise ValueError('Too many params provided')
        return params

    def _get_kwargs(self, kwargs):
        """Remove keywords that are incompatible with scipy.stats.
        """
        # TODO: think of a more general solution.
        if not self.is_conditional:
            kwargs.pop('all_samples', None)
        return kwargs


class Prior(ScipyRV):
    pass


class Model(core.ObservedMixin, ScipyRV):
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
