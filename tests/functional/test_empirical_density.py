import numpy as np
import scipy.stats as ss

import elfi.methods.empirical_density as ed

std = 5
mu = 10
nobs = 10000
N = ss.norm(loc=mu, scale=std)
X = N.rvs(size=nobs)
emp = ed.EmpiricalDensity(X)
t = np.linspace(*emp.support, 100)

def test_empirical_cdf():
    assert np.allclose(N.cdf(t), emp.cdf(t), atol=1e-2)
