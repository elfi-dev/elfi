"""Utilities for ElfiModels."""

import numpy as np


def rvs_from_distribution(*params, batch_size, distribution, size=None, random_state=None):
    """Transform the rvs method of a scipy like distribution to an operation in ELFI.

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
    return rvs


def distance_as_discrepancy(dist, *summaries, observed):
    """Evaluate a distance function with signature `dist(summaries, observed)` in ELFI."""
    summaries = np.column_stack(summaries)
    # Ensure observed are 2d
    observed = np.concatenate([np.atleast_2d(o) for o in observed], axis=1)
    try:
        d = dist(summaries, observed)
    except ValueError as e:
        raise ValueError('Incompatible data shape for the distance node. Please check '
                         'summary (XA) and observed (XB) output data dimensions. They '
                         'have to be at most 2d. Especially ensure that summary nodes '
                         'outputs 2d data even with batch_size=1. Original error message '
                         'was: {}'.format(e))
    if d.ndim == 2 and d.shape[1] == 1:
        d = d.reshape(-1)
    return d
