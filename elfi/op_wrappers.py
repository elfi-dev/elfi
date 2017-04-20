

def rvs_wrapper(*params, batch_size, distribution, size=None, random_state=None):
    """Transforms a scipy like distribution to an elfi operation

    Parameters
    ----------
    params :
        Parameters for the distribution
    batch_size : number of samples
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
        size = (batch_size, )
    else:
        size = (batch_size, ) + size

    rvs = distribution.rvs(*params, size=size, random_state=random_state)
    return dict(output=rvs)
