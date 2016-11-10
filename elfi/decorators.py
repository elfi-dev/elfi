from functools import partial

import scipy.stats as ss

from . import core
from . import distributions

__all__ = ('simulator', 'summary', 'discrepancy')


class as_op(object):
    """Decorator that turns functions into operation nodes.

    The following will produce the same results.
    >>> n0 = core.Node('n0')
    >>> def identity(x): return x
    >>> n1 = core.Operation('identity', identity, n0)
    >>> type(n1)
    <class 'elfi.core.Operation'>

    >>> @as_op(n0)
    ... def identity(x): return x
    >>> type(identity)
    <class 'elfi.core.Operation'>

    Parameters
    ----------
    *args : Any
        Arguments to be passed to the Node constructor.
    **kwargs: Any
        Keyword arguments to be passed to the Node constructor.

    Attributes
    ----------
    args : tuple
        The stored arguments.
    kwargs : dict
        The stored keyword arguments.
    _class : class
        The class of the node to construct.
    """
    _class = core.Operation

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, fun, **kwargs):
        """Construct an operation node.

        Arguments
        ---------
        fun : (Any -> Any)
            The function to be decorated.
        **kwargs : dict
            Additional arguments from inheriting decorators.

        With argumented decorators __call__ gets called once during the
        decoration process.
        """
        node = self._class(fun.__name__, fun, *self.args, **self.kwargs, **kwargs)
        node.__doc__ = fun.__doc__
        return node


class simulator(as_op):
    """Transform a function into a simulator node."""
    _class = core.Simulator

    def __init__(self, *args, observed=None, vectorized=True, **kwargs):
        self.observed = observed
        self.vectorized = vectorized
        args = map(to_elfi_distribution, args)
        super(simulator, self).__init__(*args, **kwargs)

    def __call__(self, fun, **kwargs):
        return super(simulator, self).__call__(fun, observed=self.observed,
                                               vectorized=self.vectorized, **kwargs)


# TODO: needs more work
def to_elfi_distribution(distribution):
    if isinstance(distribution, ss._distn_infrastructure.rv_frozen):
        # TODO: Doesn't handle keyword args yet
        d_args = distribution.args
        d_kwds = distribution.kwds.values()
        args = d_args or d_kwds
        return distributions.Prior(_distribution_name(distribution),
                                   distribution.dist, *args)
    else:
        return distribution


# TODO
def _distribution_name(distribution):
    return str(distribution)


class summary(as_op):
    """Transform a function into a summary node."""
    _class = core.Summary

    def __init__(self, *args, **kwargs):
        super(summary, self).__init__(*args, **kwargs)


class discrepancy(as_op):
    """Transform a function into a discrepancy node."""
    _class = core.Discrepancy

    def __init__(self, *args, **kwargs):
        super(discrepancy, self).__init__(*args, **kwargs)

