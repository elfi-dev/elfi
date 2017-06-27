import numpy as np
import scipy.stats as ss

import elfi
from elfi.methods.utils import (weighted_var, GMDistribution,
                                normalize_weights, cov2corr, corr2cov,
                                ModelPrior, numgrad)
from elfi.methods.bo.utils import stochastic_optimization, minimize
from elfi import utils


def test_stochastic_optimization():
    fun = lambda x : x**2
    bounds = ((-1, 1),)
    its = int(1e3)
    polish=True
    loc, val = stochastic_optimization(fun, bounds, its, polish)
    assert abs(loc - 0.0) < 1e-5
    assert abs(val - 0.0) < 1e-5


def test_minimize_with_known_gradient():
    fun = lambda x : x[0]**2 + (x[1]-1)**4
    grad = lambda x : np.array([2*x[0], 4*(x[1]-1)**3])
    bounds = ((-2, 2), (-2, 3))
    loc, val = minimize(fun, bounds, grad)
    assert np.isclose(val, 0, atol=0.01)
    assert np.allclose(loc, np.array([0, 1]), atol=0.02)


def test_minimize_with_approx_gradient():
    fun = lambda x : x[0]**2 + (x[1]-1)**4
    bounds = ((-2, 2), (-2, 3))
    loc, val = minimize(fun, bounds)
    assert np.isclose(val, 0, atol=0.01)
    assert np.allclose(loc, np.array([0, 1]), atol=0.02)


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

    def test_pdf(self, distribution_test):
        # 1d case
        x = [1, 2, -1]
        means = [0, 2]
        weights = normalize_weights([.4, .1])
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0]*ss.norm.pdf(x, loc=means[0]) + weights[1]*ss.norm.pdf(x, loc=means[1])
        assert np.allclose(d, d_true)

        # Test with a single observation
        # assert GMDistribution.pdf(x[0], means, weights=weights).ndim == 0

        # Distribution_test with 1d means
        distribution_test(GMDistribution, means, weights=weights)

        # 2d case
        x = [[1, 2, -1], [0,0,2]]
        means = [[0,0,0], [-1,-.2, .1]]
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0]*ss.multivariate_normal.pdf(x, mean=means[0]) + \
                 weights[1]*ss.multivariate_normal.pdf(x, mean=means[1])
        assert np.allclose(d, d_true)

        # Test with a single observation
        assert GMDistribution.pdf(x[0], means, weights=weights).ndim == 0

        # Distribution_test with 3d means
        distribution_test(GMDistribution, means, weights=weights)

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


def test_numgrad():
    assert np.allclose(numgrad(lambda x: np.log(x), 3), [1/3])
    assert np.allclose(numgrad(lambda x: np.prod(x, axis=1), [1, 3, 5]), [15, 5, 3])
    assert np.allclose(numgrad(lambda x: np.sum(x, axis=1), [1, 3, 5]), [1, 1, 1])


class TestModelPrior:

    def test_basics(self, ma2, distribution_test):
        # A 1D case
        normal = elfi.Prior('normal', 5, model=elfi.ElfiModel())
        normal_prior = ModelPrior(normal.model)
        distribution_test(normal_prior)

        # A 2D case
        prior = ModelPrior(ma2)
        distribution_test(prior)

    def test_pdf(self, ma2):
        prior = ModelPrior(ma2)
        rv = prior.rvs(size=10)
        assert np.allclose(prior.pdf(rv), np.exp(prior.logpdf(rv)))

    def test_gradient_logpdf(self, ma2):
        prior = ModelPrior(ma2)
        rv = prior.rvs(size=10)
        grads = prior.gradient_logpdf(rv)
        assert grads.shape == rv.shape
        assert np.allclose(grads, 0)

    def test_numerical_grad_logpdf(self):
        # Test gradient with a normal distribution
        loc = 2.2
        scale = 1.1
        x = np.random.rand()
        analytical_grad_logpdf = -(x - loc) / scale ** 2
        prior_node = elfi.Prior('normal', loc, scale, model=elfi.ElfiModel())
        num_grad = ModelPrior(prior_node.model).gradient_logpdf(x)
        assert np.isclose(num_grad, analytical_grad_logpdf, atol=0.01)


def test_tabulate_1d():
    arr = np.arange(4)
    grid, res = utils.tabulate(lambda x: 1 if x > 1 else 0, arr)
    expected = np.array([ 0, 0, 1, 1])

    assert np.all(grid == arr)
    assert np.all(res == expected)


def test_tabulate_2d():
    arr = np.arange(1, 4)
    grid, res = utils.tabulate(lambda x: x[0] + x[1], arr, arr)
    expected = np.array([[ 2.,  3.,  4.],
                         [ 3.,  4.,  5.],
                         [ 4.,  5.,  6.]])

    xx, yy = np.meshgrid(arr, arr)
    assert np.all(xx == grid[0])
    assert np.all(yy == grid[1])
    assert np.all(res == expected)


def test_tabulate_n():
    arr = np.arange(4)
    f = lambda x: 1 if x > 1 else 0
    g = lambda x: 0 if x > 1 else 1
    grid, res = utils.tabulate_n([f, g], arr)
    expected = [np.array([ 0, 0, 1, 1]),
                np.array([ 1, 1, 0, 0])]

    assert np.all(grid == arr)
    assert np.all(res[0] == expected[0])
    assert np.all(res[1] == expected[1])


def test_cov2corr():
    cov = np.array([[2, 0.5],
                    [0.5, 3]])
    std = np.sqrt(np.diag(cov))
    assert np.allclose(cov, corr2cov(cov2corr(cov), std))


def test_corr2cov():
    corr = np.array([[1, 0.5],
                     [0.5, 1]])
    std = np.array([2, 3])
    assert np.allclose(corr, cov2corr(corr2cov(corr, std)))
