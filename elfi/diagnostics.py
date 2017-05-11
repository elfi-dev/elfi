import numpy as np
import scipy.stats as ss

from elfi.utils import compare


def l1_dist(estimated, reference, spec, method='pdf'):
    """Compute the l1-distance between an estimation and a reference.
    
    Parameters
    ----------
    estimated :
        an object to compare
    reference :
        the second object
    spec :
        a list of tuples  of the form (min, max, number of points)
    method :
        the method to evaluate
    """
    xx, yy, est, ref = compare(estimated, reference, spec, method)
    return np.sum(abs(est - ref))


def kl_div(estimated, reference, spec):
    """Compute the Kullback-Leibler divergence between
    an estimation and a reference.
    
    Parameters
    ----------
    estimated :
        an object to compare
    reference :
        the second object
    spec :
        a list of tuples  of the form (min, max, number of points)
    """
    xx, yy, est, ref = compare(estimated, reference, spec, method='pdf')
    p = np.ravel(ref)
    q = np.ravel(est)
    return ss.entropy(p, q)

