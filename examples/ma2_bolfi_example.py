import time

import numpy as np
from abcpy.core import *
from abcpy.distributions import *
from abcpy.methods import BOLFI
from dask.dot import dot_graph
from functools import partial

import matplotlib
import matplotlib.pyplot as plt

def MA2(n, t1, t2, prng=None, latents=None):
    if latents is None:
        if prng is None:
            prng = np.random.RandomState()
        latents = prng.randn(n+2) # i.i.d. sequence ~ N(0,1)
    u = latents
    y = u[2:] + t1 * u[1:-1] + t2 * u[:-2]
    k = 0
    for i in range(10000000):
        k += 1
    print(k)
    return y

def autocov(lag, y):
    y = (y - np.mean(y, axis=1, keepdims=True)) / np.var(y, axis=1, keepdims=True)
    tau = np.sum(y[:,lag:] * y[:,:-lag], axis=1, keepdims=True)
    return tau

def distance(x, y):
    d = np.linalg.norm( np.array(x) - np.array(y), ord=2, axis=0)
    return d

def main():
    n = 100
    t1 = 0.6
    t2 = 0.2

    # Set up observed data y
    latents = np.random.randn(n+2)
    y = MA2(n, t1, t2, latents=latents)

    # Plot
    plt.figure(figsize=(11, 6));
    plt.plot(np.arange(0,n),y);
    plt.scatter(np.arange(-2,n), latents);

    # Set up the simulator
    simulator = partial(MA2, n)

    # Set up autocovariance summaries
    ac1 = partial(autocov, 1)
    ac2 = partial(autocov, 2)

    # Specify the graphical model
    t1 = Prior('t1', 'uniform', 0, 1)
    t2 = Prior('t2', 'uniform', 0, 1)
    Y = Simulator('MA2', simulator, t1, t2, observed=y)
    S1 = Summary('S1', ac1, Y)
    S2 = Summary('S2', ac2, Y)
    d = Discrepancy('d', distance, S1, S2)

    from abcpy.visualization import draw_model
    draw_model(d)

    # Specify the number of simulations and set up Bolfi sampling
    N = 10
    bounds = [(0,1), (0,1)]
    inf = BOLFI(N, d, [t1, t2], 2, bounds=bounds, sync=False)

    # Time and run the simulator in parallel
    s = time.time()
    [t1_post, t2_post] = inf.infer(1)
    print("Elapsed time %d sec" % (time.time() - s))
    print("Number of accepted samples %d" % len(t1_post))

    if len(t1_post) > 0:
        print("Posterior for t1 (mean %.2f)" % np.mean(t1_post))
        plt.hist(t1_post, bins=20)
    else:
        print("No accepted samples")

    if len(t2_post) > 0:
        print("Posterior for t2 (mean %.2f)" % np.mean(t2_post))
        plt.hist(t2_post, bins=20)
    else:
        print("No accepted samples")

if __name__ == "__main__":
    main()
