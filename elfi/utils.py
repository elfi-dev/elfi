import scipy.stats as ss
import numpy as np
import networkx as nx
import numpy as np


SCIPY_ALIASES = {
    'normal': 'norm',
    'exponential': 'expon',
    'unif': 'uniform',
    'bin': 'binom',
    'binomial': 'binom'
}


def scipy_from_str(name):
    name = name.lower()
    name = SCIPY_ALIASES.get(name, name)
    return getattr(ss, name)


def observed_name(name):
    return "_{}_observed".format(name)


def args_to_tuple(*args):
    return tuple(args)


def is_array(output):
    # Ducktyping numpy arrays
    return hasattr(output, 'shape')


# NetworkX utils


def nbunch_ancestors(G, nbunch):
    # Resolve output ancestors
    ancestors = set(nbunch)
    for node in nbunch:
        ancestors = ancestors.union(nx.ancestors(G, node))
    return ancestors


def get_sub_seed(random_state, sub_seed_index, high=2**31):
    """Returns a sub seed. The returned sub seed is unique for its index, i.e. no
    two indexes can return the same sub_seed. Same random_state will also always
    produce the same sequence.

    Parameters
    ----------
    random_state : np.random.RandomState, int
    sub_seed_index : int
    high : int
        upper limit for the range of sub seeds (exclusive)

    Returns
    -------
    int
        from interval [0, high - 1]

    Notes
    -----
    There is no guarantee how close the random_states initialized with sub_seeds may end
    up to each other. Better option is to use PRNG:s that have an advance or jump
    functions available.

    """

    if isinstance(random_state, (int, np.integer)):
        random_state = np.random.RandomState(random_state)

    if sub_seed_index >= high:
        raise ValueError("Sub seed index {} is out of range".format(sub_seed_index))

    n_unique = 0
    n_unique_required = sub_seed_index + 1
    sub_seeds = None
    seen = set()
    while n_unique != n_unique_required:
        n_draws = n_unique_required - n_unique
        sub_seeds = random_state.randint(high, size=n_draws, dtype='uint32')
        seen.update(sub_seeds)
        n_unique = len(seen)

    return sub_seeds[-1]


def grid_eval(f, spec, vectorized=True):
    """Evaluate a function on a grid.

    Parameters
    ----------
    f :
        the function to evaluate
    spec :
        a list of tuples of the form (min, max, number of points)
    vectorized : bool
        is the function vectorized? (defaults to True)
    """
    x = np.linspace(*spec[0])
    y = np.linspace(*spec[1])
    xx, yy = np.meshgrid(x, y)
    coords = np.array((xx.ravel(), yy.ravel())).T
    if vectorized:
        vals = f(coords)
    else:
        vals = np.array([f(np.array([c[0], c[1]])) for c in coords])

    vals = vals.reshape(len(x), len(y))

    return xx, yy, vals


def compare(estimated, reference, spec, method="logpdf"):
    """Evaluate the same method of two different objects on a grid.

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
    
    Returns
    -------
    If the specification is one dimensional returns a tuple
    (evaluation points, results of the estimation object, results of the reference object).
    In the two dimensional case returns a tuple (x-points, y-points, estimation results, reference results).
    """
    dim = len(spec)
    if dim == 1:
        return _compare1d(estimated, reference, spec[0], method)
    elif dim == 2:
        return _compare2d(estimated, reference, spec, method)


def _compare2d(estimated, reference, spec, method):
    _, _, est = grid_eval(getattr(estimated, method), spec)
    xx, yy, ref = grid_eval(getattr(reference, method), spec)
    return xx, yy, est, ref


def _compare1d(estimated, reference, spec, method):
    t = np.linspace(*spec)
    est = getattr(estimated, method)(t)
    ref = getattr(reference, method)(t)
    return t, est, ref

