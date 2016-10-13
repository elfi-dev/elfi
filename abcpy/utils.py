import numpy as np

from scipy.optimize import differential_evolution

def stochastic_optimization(fun, bounds, its, polish=False):
    """ Called to find the minimum of function 'fun' in 'its' iterations """
    result = differential_evolution(func=fun, bounds=bounds, maxiter=its,
                                    popsize=30, tol=0.01, mutation=(0.5, 1),
                                    recombination=0.7, disp=False,
                                    polish=polish, init='latinhypercube')
    return result.x, result.fun

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

