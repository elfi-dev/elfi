import operator

import numpy as np
from dask.delayed import delayed

from scipy.optimize import differential_evolution


# TODO: add version number to key so that resets are not confused in dask scheduler
def make_key(name, sl, version):
    """Makes the dask key for the outputs of nodes

    Parameters
    ----------
    name : string
        name of the output (e.g. node name)
    sl : slice
        data slice that is covered by this output
    version : identifier for the current version of the key
        allows one to separate results after node resets

    Returns
    -------
    a tuple key
    """
    n = slen(sl)
    if n <= 0:
        ValueError("Slice has no length")
    return (name, sl.start, n, version)


def is_elfi_key(key):
    return isinstance(key, tuple) and len(key) == 4 \
           and isinstance(key[0], str)


def get_key_slice(key):
    """Returns the corresponding slice from 'key'.
    """
    return slice(key[1], key[1] + key[2])


def get_key_name(key):
    return key[0]


def get_key_version(key):
    return key[3]


def reset_key_slice(key, new_sl):
    """Resets the slice from 'key' to 'new_sl'

    Returns
    -------
    a new key
    """
    return make_key(get_key_name(key), new_sl, get_key_version(key))


def reset_key_name(key, name):
    """Resets the name from 'key' to 'name'

    Returns
    -------
    a new key
    """
    return make_key(name, get_key_slice(key), get_key_version(key))


def get_named_item(output, item, name=None):
    """Makes a delayed object by appending "-name" to the output key name

    Parameters
    ----------
    output : delayed node output
    item : str
       item to take from the output
    name : str
       delayed key name (default: item)

    Returns
    -------
    delayed object yielding the item
    """
    name = name or item
    new_key_name = get_key_name(output.key) + '-' + str(name)
    new_key = reset_key_name(output.key, new_key_name)
    return delayed(operator.getitem)(output, item, dask_key_name=new_key)


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


# Fixme: the two below ones seem quite specialized. Should they be moved somewhere else?
def stochastic_optimization(fun, bounds, its, polish=False):
    """ Called to find the minimum of function 'fun' in 'its' iterations """
    result = differential_evolution(func=fun, bounds=bounds, maxiter=its,
                                    popsize=30, tol=0.01, mutation=(0.5, 1),
                                    recombination=0.7, disp=False,
                                    polish=polish, init='latinhypercube')
    return result.x, result.fun


def weighted_var(data, weights):
    """Weighted variance.

    Parameters
    ----------
    data : np.array of shape (n, m)
    weights : 1d np.array of shape (n,)

    Returns
    -------
    np.array of shape (m,)
    """
    weighted_mean = np.average(data, weights=weights, axis=0)
    return np.average((data - weighted_mean)**2, weights=weights, axis=0)
