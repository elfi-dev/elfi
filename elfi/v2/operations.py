from elfi.v2.utils import splen


def rvs_operation(*params, span, distribution, random_state, size):
    """Transforms a scipy like distribution to an elfi operation

    Parameters
    ----------
    params :
        Parameters for the distribution
    span : tuple
    distribution : scipy-like distribution object
    random_state : RandomState object
    size : tuple
        Size of a single datum from the distribution.

    Returns
    -------
    random variates from the distribution

    Notes
    -----
    Used internally by the RandomVariable to wrap distributions for the framework.

    """
    size = (splen(span), ) + size
    return distribution.rvs(*params, size=size, random_state=random_state)