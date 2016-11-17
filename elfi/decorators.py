from functools import partial

from elfi.core import vectorize_simulator
from elfi.core import vectorize_summary
from elfi.core import vectorize_discrepancy


def as_vectorized_simulator(simulator):
    """Vectorizes a sequential simulator.

    Preliminary implementation, interface subject to change.

    Parameters
    ----------
    simulator : sequential simulator function

    Returns
    -------
    simulator : vectorized simulator function
    """
    return partial(vectorize_simulator, simulator)


def as_vectorized_summary(summary):
    """Vectorizes a sequential summary operation.

    Preliminary implementation, interface subject to change.

    Parameters
    ----------
    summary : sequential summary function

    Returns
    -------
    summary : vectorized summary function
    """
    return partial(vectorize_summary, summary)

def as_vectorized_discrepancy(discrepancy):
    """Vectorizes a sequential discrepancy operation.

    Preliminary implementation, interface subject to change.

    Parameters
    ----------
    discrepancy : sequential discrepancy function

    Returns
    -------
    discrepancy : vectorized discrepancy function
    """
    return partial(vectorize_discrepancy, discrepancy)


