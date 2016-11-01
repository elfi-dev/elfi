
from elfi.core import *
from elfi.distributions import *
import numpy as np
import numpy.random as npr

# Define some summary statistic and discrepancy functions
def summary(x):
    return np.median(x, axis=np.ndim(x)-1, keepdims=True)

def summary2(x):
    return np.var(x, axis=np.ndim(x)-1, keepdims=True)

def discrepancy(generated, observed):
    return np.linalg.norm(np.array(generated) - np.array(observed), ord=1, axis=0)

# Generate parameters
N = 100
n = 100

# True parameter and random data
mu_0 = 3
y_0 = npr.normal(mu_0, mu_0, size=n)

# Build the ABC network
mu = Prior('mu', 'expon', 5)
Y = Model('Y', 'normal', mu, mu, observed=y_0)
S = Summary('S', summary, Y)
S2 = Summary('S2', summary2, Y)
d = Discrepancy('d', discrepancy, S, S2)
t = Threshold('t', 3, d)

print(mu.generate(N).compute())
print(mu[0:N].compute())
print(Y.generate(N).compute())
print(S.generate(N).compute())
print(S.observed)
print(S2.generate(N).compute())
print(S2.observed)
print(d.generate(N).compute())
print(t.generate(N).compute())

print('\nSample')
print(mu[0:N].compute()[t[0:N].compute()])


print('\nTesting generating with_values\n')
print(d.generate(10, with_values={'mu': np.array([4,]*10, ndmin=2).T}).compute())
