

def rvs_operation(*params, n, distribution, size=None, random_state=None):
    """Transforms a scipy like distribution to an elfi operation

    Parameters
    ----------
    params :
        Parameters for the distribution
    n : number of samples
    distribution : scipy-like distribution object
    size : tuple
        Size of a single datum from the distribution.
    random_state : RandomState object or None

    Returns
    -------
    random variates from the distribution

    Notes
    -----
    Used internally by the RandomVariable to wrap distributions for the framework.

    """

    if size is None:
        size = (n, )
    else:
        size = (n, ) + size

    print(params)

    rvs = distribution.rvs(*params, size=size, random_state=random_state)
    return dict(output=rvs)