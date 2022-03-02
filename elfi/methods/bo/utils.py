"""Utilities for Bayesian optimization."""

import numpy as np
import scipy.optimize
from scipy.optimize import differential_evolution


# TODO: remove or combine to minimize
def stochastic_optimization(fun, bounds, maxiter=1000, polish=True, seed=0):
    """Find the minimum of function 'fun' in 'maxiter' iterations.

    Parameters
    ----------
    fun : callable
        Function to minimize.
    bounds : list of tuples
        Bounds for each parameter.
    maxiter : int, optional
        Maximum number of iterations.
    polish : bool, optional
        Whether to "polish" the result.
    seed : int, optional

    See scipy.optimize.differential_evolution.

    Returns
    -------
    tuple of the found coordinates of minimum and the corresponding value.

    """
    def fun_1d(x):
        return fun(x).ravel()

    result = differential_evolution(
        func=fun_1d, bounds=bounds, maxiter=maxiter,
        polish=polish, init='latinhypercube', seed=seed)
    return result.x, result.fun


def minimize(fun,
             bounds,
             method='L-BFGS-B',
             constraints=None,
             grad=None,
             prior=None,
             n_start_points=10,
             maxiter=1000,
             random_state=None):
    """Find the minimum of function 'fun'.

    Parameters
    ----------
    fun : callable
        Function to minimize.
    bounds : list of tuples
        Bounds for each parameter.
    method : str or callable, optional
        Minimizer method to use, defaults to L-BFGS-B.
    constraints : {Constraint, dict} or List of {Constraint, dict}, optional
        Constraints definition (only for COBLYA, SLSQP and trust-constr).
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

    # Run the optimisation from each initialization point.
    locs = []
    vals = np.empty(n_start_points)
    for i in range(n_start_points):
        result = scipy.optimize.minimize(fun, start_points[i, :],
                                         method=method, jac=grad,
                                         bounds=bounds, constraints=constraints)
        locs.append(result['x'])
        vals[i] = result['fun']

    # Return the optimal case.
    ind_min = np.argmin(vals)
    locs_out = locs[ind_min]
    for i in range(ndim):
        locs_out[i] = np.clip(locs_out[i], *bounds[i])

    return locs[ind_min], vals[ind_min]


class CostFunction:
    """Convenience class for modelling acquisition costs."""

    def __init__(self, function, gradient, scale=1):
        """Initialise CostFunction.

        Parameters
        ----------
        function : callable
            Function that returns cost function value.
        gradient : callable
            Function that returns cost function gradient.
        scale : float, optional
            Cost function is multiplied with scale.

        """
        self.function = function
        self.gradient = gradient
        self.scale = scale

    def evaluate(self, x):
        """Return cost function value evaluated at x.

        Parameters
        ----------
        x : np.ndarray, shape: (input_dim,) or (n, input_dim)

        Returns
        -------
        np.ndarray, shape: (n, 1)

        """
        x = np.atleast_2d(x)
        n, input_dim = x.shape
        return self.scale * self.function(x).reshape(n, 1)

    def evaluate_gradient(self, x):
        """Return cost function gradient evaluated at x.

        Parameters
        ----------
        x : np.ndarray, shape: (input_dim,) or (n, input_dim)

        Returns
        -------
        np.ndarray, shape: (n, input_dim)

        """
        x = np.atleast_2d(x)
        n, input_dim = x.shape
        return self.scale * self.gradient(x).reshape(n, input_dim)
