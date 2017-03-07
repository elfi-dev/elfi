from functools import partial
import pytest
import numpy as np

from elfi import mcmc


# construct a covariance matrix and calculate the precision matrix
n = 5
true_cov = np.random.rand(n,n) * 0.5
true_cov += true_cov.T
true_cov += np.eye(n)
prec = np.linalg.inv(true_cov)


# log multivariate Gaussian pdf
def log_pdf(x):
    return -0.5 * x.dot(prec).dot(x)


# gradient of log multivariate Gaussian pdf
def grad_log_pdf(x):
    return -x.dot(prec)


class TestMetropolis():
    def test_metropolis(self):
        n_samples = 1000
        x_init = np.random.rand(n)
        sigma = np.ones(n) * 0.1
        samples = mcmc.metropolis(n_samples, x_init, log_pdf, sigma)
        assert samples.shape == (n_samples, n)
        # cov = np.cov(samples.T)
        # assert np.allclose(cov, true_cov, atol=0.3)


class TestNUTS():
    def test_nuts(self):
        n_samples = 20000
        n_adapt = 3000
        x_init = np.random.rand(n)
        samples = mcmc.nuts(n_samples, x_init, log_pdf, grad_log_pdf, n_adapt=n_adapt)
        assert samples.shape == (n_samples, n)
        cov = np.cov(samples.T)
        assert np.allclose(cov, true_cov, atol=0.2)

