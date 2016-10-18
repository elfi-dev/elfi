import numpy as np


def MA2(n, N, t1, t2, prng=None, latents=None):
    if latents is None:
        if prng is None:
            prng = np.random.RandomState()
        latents = prng.randn(N,n+2) # i.i.d. sequence ~ N(0,1)
    u = np.atleast_2d(latents)
    y = u[:,2:] + t1 * u[:,1:-1] + t2 * u[:,:-2]
    return y


def autocov(lag, y):
    y = (y - np.mean(y, axis=1, keepdims=True)) / np.var(y, axis=1, keepdims=True)
    tau = np.sum(y[:,lag:] * y[:,:-lag], axis=1, keepdims=True)
    return tau


def distance(x, y):
    d = np.linalg.norm( np.array(x) - np.array(y), ord=2, axis=0)
    return d

