import numpy as np
import numdifftools
from scipy.optimize import differential_evolution, fmin_l_bfgs_b


# TODO: remove or combine to minimize
def stochastic_optimization(fun, bounds, maxiter=1000, polish=True, seed=0):
    """ Called to find the minimum of function 'fun' in 'maxiter' iterations """
    result = differential_evolution(func=fun, bounds=bounds, maxiter=maxiter,
                                    polish=polish, init='latinhypercube', seed=seed)
    return result.x, result.fun


def minimize(fun, grad, bounds, prior=None, n_start_points=10, maxiter=1000, random_state=None):
    """ Called to find the minimum of function 'fun'.
    
    Parameters
    ----------
    fun : callable
        Function to minimize.
    grad : callable
        Gradient of fun.
    bounds : list of tuples
        Bounds for each parameter.
    prior
        Used for sampling initialization points. If None, samples uniformly.
    n_start_points : int, optional
        Number of initialization points.
    maxiter : int, optional
        Maximum number of iterations.
    random_state : np.random.RandomState, optional
        Used only if no elfi.Priors given.
    
    Returns
    -------
    tuple of the found coordinates of minimum and the corresponding value.
    """
    ndim = len(bounds)
    start_points = np.empty((n_start_points, ndim))

    # TODO: use same prior as the bo.acquisition.UniformAcquisition
    if prior is None:
        # Sample initial points uniformly within bounds
        random_state = random_state or np.random.RandomState()
        for i in range(ndim):
            start_points[:, i] = random_state.uniform(*bounds[i], n_start_points)
    else:
        start_points = prior.rvs(n_start_points)
        for i in range(ndim):
            start_points[:, i] = np.clip(start_points[:, i], *bounds[i])

    locs = []
    vals = np.empty(n_start_points)

    # Run optimization from each initialization point
    for i in range(n_start_points):
        result = fmin_l_bfgs_b(fun, start_points[i, :], fprime=grad, bounds=bounds, maxiter=maxiter)
        locs.append(result[0])
        vals[i] = result[1]

    # Return the optimal case
    ind_min = np.argmin(vals)
    return locs[ind_min], vals[ind_min]

