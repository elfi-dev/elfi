import numpy as np
from scipy.optimize import differential_evolution, fmin_l_bfgs_b


# TODO: remove or combine to minimize
def stochastic_optimization(fun, bounds, maxiter=1000, polish=True, seed=0):
    """ Called to find the minimum of function 'fun' in 'maxiter' iterations """
    result = differential_evolution(func=fun, bounds=bounds, maxiter=maxiter,
                                    polish=polish, init='latinhypercube', seed=seed)
    return result.x, result.fun


# TODO: allow argument for specifying the optimization algorithm
def minimize(fun, bounds, grad=None, prior=None, n_start_points=10, maxiter=1000, random_state=None):
    """ Called to find the minimum of function 'fun'.
    
    Parameters
    ----------
    fun : callable
        Function to minimize.
    bounds : list of tuples
        Bounds for each parameter.
    grad : callable
        Gradient of fun or None.
    prior : scipy-like distribution object
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

    if prior is None:
        # Sample initial points uniformly within bounds
        # TODO: combine with the the bo.acquisition.UniformAcquisition method?
        random_state = random_state or np.random
        for i in range(ndim):
            start_points[:, i] = random_state.uniform(*bounds[i], n_start_points)
    else:
        start_points = prior.rvs(n_start_points, random_state=random_state)
        if len(start_points.shape) == 1:
            # Add possibly missing dimension when ndim=1
            start_points = start_points[:, None]
        for i in range(ndim):
            start_points[:, i] = np.clip(start_points[:, i], *bounds[i])

    locs = []
    vals = np.empty(n_start_points)

    # Run optimization from each initialization point
    for i in range(n_start_points):
        if grad is not None:
            result = fmin_l_bfgs_b(fun, start_points[i, :], fprime=grad, bounds=bounds, maxiter=maxiter)
        else:
            result = fmin_l_bfgs_b(fun, start_points[i, :], approx_grad=True, bounds=bounds, maxiter=maxiter)
        locs.append(result[0])
        vals[i] = result[1]

    # Return the optimal case
    ind_min = np.argmin(vals)
    return locs[ind_min], vals[ind_min]

