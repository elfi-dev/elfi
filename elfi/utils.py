
from scipy.optimize import differential_evolution


# Fixme: this seems quite specialized. Should it be moved to where it is used?
def stochastic_optimization(fun, bounds, its, polish=False):
    """ Called to find the minimum of function 'fun' in 'its' iterations """
    result = differential_evolution(func=fun, bounds=bounds, maxiter=its,
                                    popsize=30, tol=0.01, mutation=(0.5, 1),
                                    recombination=0.7, disp=False,
                                    polish=polish, init='latinhypercube')
    return result.x, result.fun


def to_slice(item):
    """Converts item specifier to slice

    Currently handles only either slices or integers

    Parameters
    ----------
    item
       The things that is passed to `__getitem__` from object[item]

    Returns
    -------
    slice

    """
    if not isinstance(item, slice):
        item = slice(item, item + 1)
    return item


def slice_intersect(sl1, sl2=None, offset=0):
    sl2 = sl2 or sl1
    intsect_sl = slice(max(sl2.start, sl1.start) - offset, min(sl2.stop, sl1.stop) - offset)
    if intsect_sl.stop - intsect_sl.start <= 0:
        intsect_sl = slice(offset, offset)
    return intsect_sl


def slen(sl):
    return sl.stop - sl.start