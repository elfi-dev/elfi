import numpy as np
import scipy.stats as ss

from elfi.utils import compare


def l1_dist(estimated, reference, spec, method='pdf'):
    xx, yy, est, ref = compare(estimated, reference, spec, method)
    return np.sum(abs(est - ref))


def kl_div(estimated, reference, spec):
    xx, yy, est, ref = compare(estimated, reference, spec, method='pdf')
    p = np.ravel(ref)
    q = np.ravel(est)
    return ss.entropy(p, q)

