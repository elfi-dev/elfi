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


def KLIEP(x, y, sigma, prior, n=100, epsilon=0.001, max_iter=100):
    """Kullback-Leibler Importance Estimation Procedure for ratio estimation.

    Parameters
    ----------
    x : array
        Sample from the nominator distribution.
    y : sample
        Sample from the denominator distribution.
    sigma : float
        RBF kernel sigma.
    prior : distribution object
        Determines RBF basis means.
    n : int
        Number of RBF basis functions.
    epsilon : float
        Parameter determining speed of gradient descent.

    Returns
    -------
    Ratio-estimate of two distributions

    """
    theta = prior.rvs(size=n)
    y_len = y.shape[0]

    A = np.array([[RBF(i, j, sigma) for j in theta] for i in x])
    b = np.sum(np.array([[RBF(j, i, sigma) for j in y] for i in theta]), axis=1) / y_len
    alpha = np.random.uniform(size=n, low=0, high=0.2)
    for i in np.arange(max_iter):
        alpha = alpha + epsilon * np.dot(A.T, (1 / (np.dot(A, alpha))))
        alpha = np.maximum(0, alpha + (1 - np.dot(b.T, alpha)) * b / np.dot(b.T, b))
        alpha = alpha / np.dot(b.T, alpha)

    def w(x):
        return np.dot(np.array([[RBF(i, j, sigma) for j in theta] for i in x]), alpha)
        # return alpha * np.exp(-0.5 * np.linalg.norm(x - theta) ** 2 / sigma / sigma)

    return w


def RBF(x, x0, sigma):
    """N-D RBF basis-function with equal scale-parameter for every dim."""
    return np.exp(-0.5 * np.linalg.norm(x - x0) ** 2 / sigma / sigma)
