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
    ret = 0
    for i in range(kern_centers.shape[0]):
        sqdist = sum((kern_centers[i,:] - point) ** 2)
        ret += kern_ampl * np.exp(-sqdist / kern_scale)
    return ret

