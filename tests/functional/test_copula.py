import numpy as np
import pytest
import scipy.stats as ss

from elfi.methods.copula import GaussianCopula
from elfi.diagnostics import kl_div

# Test data
cov = np.array([[2, 0.5], [0.5, 3]])
mean = np.array([1, 5])
nobs = 2000
N = ss.multivariate_normal(mean=mean, cov=cov)
X = N.rvs(size=nobs)
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)

# Empirical Gaussian copula
cov_est = np.cov(X, rowvar=False)
gc = GaussianCopula(cov=cov_est, marginal_samples=X)

def test_gaussian_copula():
    n_points = 30
    eps = 1e-2
    spec = [(mins[0], maxs[0], n_points),
            (mins[1], maxs[1], n_points)]
    
    assert kl_div(gc, N, spec) < eps

def test_gaussian_copula_cov():
    gc = GaussianCopula(cov, [ss.norm(0, 1), ss.norm(0, 1)])
    Z = gc.rvs(10000)
    cov_est = np.cov(X, rowvar=False)

    assert np.allclose(cov, cov_est)
    

