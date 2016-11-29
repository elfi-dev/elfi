import numpy as np

"""Example implementation of the MA2 model
"""

# TODO: add tests

def MA2(n_obs, t1, t2, n_sim=1, prng=None, latents=None):
    if latents is None:
        if prng is None:
            prng = np.random.RandomState()
        latents = prng.randn(n_sim, n_obs+2) # i.i.d. sequence ~ N(0,1)
    u = np.atleast_2d(latents)
    y = u[:,2:] + t1 * u[:,1:-1] + t2 * u[:,:-2]
    return y


def autocov(lag, x):
    """Autocovariance assuming a (weak) univariate stationary process
    with realizations in rows
    """
    mu = np.mean(x, axis=1, keepdims=True)
    C = np.mean(x[:,lag:] * x[:,:-lag], axis=1, keepdims=True) - mu**2
    return C


def distance(x, y):
    d = np.linalg.norm( np.array(x) - np.array(y), ord=2, axis=0)
    return d

