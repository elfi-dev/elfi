import time

import numpy as np
import dask
from elfi.core import *
from elfi.distributions import *
from examples.ma2 import MA2, autocov, distance
from distributed import Client
from functools import partial

# Distributed setup
# client = Client()
# dask.set_options(get=client.get)

# Setup the model
n = 100
t1 = 0.6
t2 = 0.2

# Set up observed data y
latents = np.random.randn(n+2)
y = MA2(n, 1, t1, t2, latents=latents)

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

dists = d.generate(4, with_values={'t1': .4, 't2': .1}, batch_size=1)
print(dists.compute())