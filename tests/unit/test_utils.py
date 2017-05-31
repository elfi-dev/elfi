import numpy as np
import scipy.stats as ss

from elfi.methods.utils import weighted_var, GMDistribution, normalize_weights
from elfi.methods.bo.utils import stochastic_optimization, minimize, numerical_grad_logpdf


def test_stochastic_optimization():
    fun = lambda x : x**2
    bounds = ((-1, 1),)
    its = int(1e3)
    polish=True
    loc, val = stochastic_optimization(fun, bounds, its, polish)
    assert abs(loc - 0.0) < 1e-5
    assert abs(val - 0.0) < 1e-5


def test_minimize():
    fun = lambda x : x[0]**2 + (x[1]-1)**4
    grad = lambda x : np.array([2*x[0], 4*(x[1]-1)**3])
    bounds = ((-2, 2), (-2, 3))
    priors = [None, None]
    loc, val = minimize(fun, grad, bounds, priors)
    assert np.isclose(val, 0, atol=0.01)
    assert np.allclose(loc, np.array([0, 1]), atol=0.01)


def test_numerical_grad_logpdf():
    dist = ss.norm
    loc = 2.2
    scale = 1.1
    x = np.random.rand()
    grad_logpdf = -(x-loc)/scale**2
    num_grad = numerical_grad_logpdf(x, loc, scale, distribution=dist)
    assert np.isclose(grad_logpdf, num_grad, atol=0.01)


def test_weighted_var():
    # 1d case
    std = .3
    x = np.random.RandomState(12345).normal(-2, std, size=1000)
    w = np.array([1] * len(x))
    assert (weighted_var(x, w) - std) < .1

    # 2d case
    cov = [[.5, 0], [0, 3.2]]
    x = np.random.RandomState(12345).multivariate_normal([1,2], cov, size=1000)
    w = np.array([1] * len(x))
    assert np.linalg.norm(weighted_var(x, w) - np.diag(cov)) < .1


class TestGMDistribution:

    def test_pdf(self):
        # 1d case
        x = [1, 2, -1]
        means = [0, 2]
        weights = normalize_weights([.4, .1])
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0]*ss.norm.pdf(x, loc=means[0]) + weights[1]*ss.norm.pdf(x, loc=means[1])
        assert np.allclose(d, d_true)

        # 2d case
        x = [[1, 2, -1], [0,0,2]]
        means = [[0,0,0], [-1,-.2, .1]]
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0]*ss.multivariate_normal.pdf(x, mean=means[0]) + \
                 weights[1]*ss.multivariate_normal.pdf(x, mean=means[1])
        assert np.allclose(d, d_true)

    def test_rvs(self):
        means = [[1000, 3], [-1000, -3]]
        weights = [.3, .7]
        N = 10000
        random = np.random.RandomState(12042017)
        rvs = GMDistribution.rvs(means, weights=weights, size=N, random_state=random)
        rvs = rvs[rvs[:,0] < 0, :]

        # Test correct proportion of samples near the second mode
        assert np.abs(len(rvs)/N - .7) < .01

        # Test that the mean of the second mode is correct
        assert np.abs(np.mean(rvs[:,1]) + 3) < .1
