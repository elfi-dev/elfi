import numpy as np
import numdifftools

from scipy.optimize import differential_evolution, fmin_l_bfgs_b


def approx_second_partial_derivative(fun, x0, dim, h, bounds):
    """
        Approximates the second derivative of function 'fun' at 'x0'
        in dimension 'dim'. If sampling location is near the bounds,
        uses a symmetric approximation.
    """
    val = fun(x0)
    d = np.zeros(len(x0))
    d[dim] = 1.0
    if (x0 + h*d)[dim] > bounds[dim][1]:
        # At upper edge, using symmetric approximation
        val_m = fun(x0 - h*d)
        val_p = val_m
    elif (x0 - h*d)[dim] < bounds[dim][0]:
        # At upper edge, using symmetric approximation
        val_p = fun(x0 + h*d)
        val_m = val_p
    else:
        val_p = fun(x0 + h*d)
        val_m = fun(x0 - h*d)
    return (val_p - 2*val + val_m) / (h ** 2)


def sum_of_rbf_kernels(point, kern_centers, kern_ampl, kern_scale):
    """
        Calculates the sum of kernel weights at 'point' given that
        there is one RBF kernel at each 'kern_center' and they
        all have same amplitudes and scales.

        type(point) = np.array_1d
        type(kern_certers) = np.array_2d (centers on rows)
    """
    if kern_scale <= 0:
        raise ValueError("RBF kernel scale must be positive"
                         "(was: %.2f)" % (kern_scale))
    if kern_ampl < 0:
        raise ValueError("RBF kernel amplitude must not be negative"
                         "(was: %.2f)" % (kern_ampl))
    if kern_ampl == 0:
        return 0
    if len(kern_centers) == 0:
        return 0
    if kern_centers.shape[1] != point.shape[0]:
        raise ValueError("kern_centers shape must match point shape")
    ret = 0
    for i in range(kern_centers.shape[0]):
        sqdist = sum((kern_centers[i,:] - point) ** 2)
        ret += kern_ampl * np.exp(-sqdist / kern_scale)
    return ret


def stochastic_optimization(fun, bounds, maxiter=1000, polish=True, seed=0):
    """ Called to find the minimum of function 'fun' in 'maxiter' iterations """
    result = differential_evolution(func=fun, bounds=bounds, maxiter=maxiter,
                                    polish=polish, init='latinhypercube', seed=seed)
    return result.x, result.fun


def minimize(fun, grad, bounds, priors, n_inits=10, maxiter=1000, random_state=None):
    """ Called to find the minimum of function 'fun'.
    
    Parameters
    ----------
    fun : callable
        Function to minimize.
    grad : callable
        Gradient of fun.
    bounds : list of tuples
        Bounds for each parameter.
    priors : list of elfi.Priors, or list of Nones
        Used for sampling initialization points. If Nones, sample uniformly. 
    n_inits : int, optional
        Number of initialization points.
    maxiter : int, optional
        Maximum number of iterations.
    random_state : np.random.RandomState, optional
        Used only if no elfi.Priors given.
    
    Returns
    -------
    tuple of the found coordinates of minimum and the corresponding value.
    """
    inits = np.empty((n_inits, len(priors)))

    if priors[0] is None:
        # Sample initial points uniformly within bounds
        random_state = random_state or np.random.RandomState()
        for ii in range(len(priors)):
            inits[:, ii] = random_state.uniform(*bounds[ii], n_inits)

    else:
        # Sample priors for initialization points
        prior_names = [p.name for p in priors]
        inits_dict = priors[0].model.generate(n_inits, outputs=prior_names)
        for ii, n in enumerate(prior_names):
            inits[:, ii] = inits_dict[n]
            inits[:, ii] = np.clip(inits[:, ii], *bounds[ii])

    locs = []
    vals = np.empty(n_inits)

    # Run optimization from each initialization point
    for ii in range(n_inits):
        result = fmin_l_bfgs_b(fun, inits[ii, :], fprime=grad, bounds=bounds, maxiter=maxiter)
        locs.append(result[0])
        vals[ii] = result[1]

    # Return the optimal case
    ind_min = np.argmin(vals)
    return locs[ind_min], vals[ind_min]


def numerical_grad_logpdf(x, *params, distribution=None, **kwargs):
    """Gradient of the log of the probability density function at x.
    
    Approximated numerically.

    Parameters
    ----------
    x : array_like
        points where to evaluate the gradient
    param1, param2, ... : array_like
        parameters of the model
    distribution : ScipyLikeDistribution or a distribution from Scipy

    Returns
    -------
    grad_logpdf : ndarray
       Gradient of the log of the probability density function evaluated at x
    """

    # due to the common scenario of logpdf(x) = -inf, multiple confusing warnings could be generated
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')
        if np.isinf(distribution.logpdf(x, *params, **kwargs)):
            grad = np.zeros_like(x)  # logpdf = -inf => grad = 0
        else:
            grad = numdifftools.Gradient(distribution.logpdf)(x, *params, **kwargs)
            grad = np.where(np.isnan(grad), 0, grad)

    return grad
