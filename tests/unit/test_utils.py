import json
from collections import OrderedDict

import numpy as np
import scipy.stats as ss

import elfi
from elfi.examples.ma2 import get_model
from elfi.methods.bo.utils import CostFunction, minimize, stochastic_optimization
from elfi.methods.density_ratio_estimation import DensityRatioEstimation
from elfi.methods.utils import (GMDistribution, normalize_weights, numgrad, numpy_to_python_type,
                                sample_object_to_dict, weighted_sample_quantile, weighted_var)
from elfi.model.extensions import ModelPrior


def test_stochastic_optimization():
    def fun(x):
        return x**2

    bounds = ((-1, 1), )
    its = int(1e3)
    polish = True
    loc, val = stochastic_optimization(fun, bounds, its, polish)
    assert abs(loc - 0.0) < 1e-5
    assert abs(val - 0.0) < 1e-5


def test_minimize_with_known_gradient():
    def fun(x):
        return x[0]**2 + (x[1] - 1)**4

    def grad(x):
        return np.array([2 * x[0], 4 * (x[1] - 1)**3])

    bounds = ((-2, 2), (-2, 3))
    loc, val = minimize(fun, bounds, grad=grad)
    assert np.isclose(val, 0, atol=0.01)
    assert np.allclose(loc, np.array([0, 1]), atol=0.02)


def test_minimize_with_approx_gradient():
    def fun(x):
        return x[0]**2 + (x[1] - 1)**4

    bounds = ((-2, 2), (-2, 3))
    loc, val = minimize(fun, bounds)
    assert np.isclose(val, 0, atol=0.01)
    assert np.allclose(loc, np.array([0, 1]), atol=0.02)


def test_minimize_with_constraints():
    def fun(x):
        return x[0]**2 + (x[1] - 1)**4

    bounds = ((-2, 2), (-2, 3))
    # Test constraint y >= x
    constraints = ({'type': 'ineq',
                    'fun': lambda x : x[1] - x[0]})
    loc, val = minimize(fun, bounds, constraints=constraints, method='SLSQP')
    assert np.isclose(val, 0, atol=0.01)
    assert np.allclose(loc, np.array([0, 1]), atol=0.02)


def test_weighted_sample_quantile():
    x = np.arange(11)
    alpha_q1 = weighted_sample_quantile(x=x, alpha=0.50)
    assert alpha_q1 == x[5]

    weights = np.array((0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
    alpha_q2 = weighted_sample_quantile(x=x, alpha=0.50, weights=weights)
    assert alpha_q2 == x[8]

    alpha_q3 = weighted_sample_quantile(x=x, alpha=3/11)
    alpha_q4 = weighted_sample_quantile(x=np.flip(x), alpha=3/11)
    assert alpha_q3 == alpha_q4


def test_weighted_var():
    # 1d case
    std = .3
    x = np.random.RandomState(12345).normal(-2, std, size=1000)
    w = np.array([1] * len(x))
    assert (weighted_var(x, w) - std) < .1

    # 2d case
    cov = [[.5, 0], [0, 3.2]]
    x = np.random.RandomState(12345).multivariate_normal([1, 2], cov, size=1000)
    w = np.array([1] * len(x))
    assert np.linalg.norm(weighted_var(x, w) - np.diag(cov)) < .1


class TestGMDistribution:
    def test_pdf(self, distribution_test):
        # 1d case
        x = [1, 2, -1]
        means = [0, 2]
        weights = normalize_weights([.4, .1])
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0] * ss.norm.pdf(x, loc=means[0]) + \
            weights[1] * ss.norm.pdf(x, loc=means[1])
        assert np.allclose(d, d_true)

        # Test with a single observation
        # assert GMDistribution.pdf(x[0], means, weights=weights).ndim == 0

        # Distribution_test with 1d means
        distribution_test(GMDistribution, means, weights=weights)

        # 2d case
        x = [[1, 2, -1], [0, 0, 2]]
        means = [[0, 0, 0], [-1, -.2, .1]]
        d = GMDistribution.pdf(x, means, weights=weights)
        d_true = weights[0] * ss.multivariate_normal.pdf(x, mean=means[0]) + \
            weights[1] * ss.multivariate_normal.pdf(x, mean=means[1])
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
        rvs = rvs[rvs[:, 0] < 0, :]

        # Test correct proportion of samples near the second mode
        assert np.abs(len(rvs) / N - .7) < .01

        # Test that the mean of the second mode is correct
        assert np.abs(np.mean(rvs[:, 1]) + 3) < .1

    def test_rvs_prior_ok(self):
        means = [0.8, 0.5]
        weights = [.3, .7]
        N = 10000
        prior_logpdf = ss.uniform(0, 1).logpdf
        rvs = GMDistribution.rvs(means, weights=weights, size=N, prior_logpdf=prior_logpdf)

        # Ensure prior pdf > 0 for all samples
        assert np.all(np.isfinite(prior_logpdf(rvs)))


def test_numgrad():
    assert np.allclose(numgrad(lambda x: np.log(x), 3), [1 / 3])
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
        analytical_grad_logpdf = -(x - loc) / scale**2
        prior_node = elfi.Prior('normal', loc, scale, model=elfi.ElfiModel())
        num_grad = ModelPrior(prior_node.model).gradient_logpdf(x)
        assert np.isclose(num_grad, analytical_grad_logpdf, atol=0.01)


def test_sample_object_to_dict():
    data_rej = OrderedDict()
    data_smc = OrderedDict()
    m = get_model(n_obs=100, true_params=[.6, .2])
    batch_size, n = 1, 2
    schedule = [0.7, 0.2, 0.05]
    rej = elfi.Rejection(m['d'], batch_size=batch_size)
    res_rej = rej.sample(n, threshold=0.1)
    smc = elfi.SMC(m['d'], batch_size=batch_size)
    res_smc = smc.sample(n, schedule)
    sample_object_to_dict(data_rej, res_rej)
    sample_object_to_dict(data_smc, res_smc, skip='populations')
    assert any(x not in data_rej for x in ['meta', 'output']) is True
    assert any(x not in data_smc for x in ['meta', 'output', 'populations']) is True


def test_numpy_to_python_type():
    data = dict(a=np.array([1, 2, 3, 4]), b=np.uint(5), c=np.float_(10),
                d=dict(a=np.array([0, 9, 8, 7]), b=np.uint(15), c=np.float_(12)))
    numpy_to_python_type(data)

    # checking that our objects are jsonable is enough to be sure that numpy_to_python_type
    # function works fine
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except:
            return False

    assert is_jsonable(data) is True


class TestDensityRatioEstimation:
    def test_shapes(self):
        N = 100
        n = 50
        x = ss.multivariate_normal.rvs(size=N, mean=[0, 0], cov=np.diag([1, 1]))
        y = ss.multivariate_normal.rvs(size=N, mean=[0, 0], cov=np.diag([2, 2]))
        densratio = DensityRatioEstimation(n=n)
        densratio.fit(x, y, sigma=1.0)
        A = densratio._compute_A(x, 1.0)
        b, b_normalized = densratio._compute_b(y, 1.0)

        assert A.shape[0] == N
        assert A.shape[1] == n
        assert b.shape[0] == n
        assert b_normalized.shape[0] == n
        assert densratio.w([0, 0]).shape == (1,)

    def test_consistency(self):
        N = 100
        n = 50
        x = ss.multivariate_normal.rvs(size=N, mean=[0, 0], cov=np.diag([1, 1]))
        y = ss.multivariate_normal.rvs(size=N, mean=[0, 0], cov=np.diag([2, 2]))
        densratio1 = DensityRatioEstimation(n=n, optimize=True)
        densratio2 = DensityRatioEstimation(n=n, optimize=True)
        densratio3 = DensityRatioEstimation(n=n, optimize=True)
        densratio4 = DensityRatioEstimation(n=n)

        densratio1.fit(x, y, sigma=list([1.0, 2.0]))
        densratio2.fit(x, y, sigma=list([1.0, 2.0]))


        sigma_list = list([0.1, 2.0, 10])
        densratio3.fit(x, y, sigma=sigma_list)

        weights_x = np.ones(N)
        weights_x[:5] = 10.0
        weights_y = np.ones(N)
        densratio4.fit(x, y, sigma=1.0, weights_x=weights_x, weights_y=weights_y)

        assert densratio1.sigma == 1.0
        assert densratio2.sigma == 1.0
        assert densratio1.w([0, 0]) == densratio2.w([0, 0])
        assert densratio3.sigma in sigma_list
        assert densratio1.w([0, 0]) != densratio3.w([0, 0])
        assert densratio1.w([0, 0]) != densratio4.w([0, 0])

    def test_ratio_estimation(self):
        N = 1000
        x = ss.norm.rvs(size=N, loc=1, scale=0.1, random_state=123)
        y = ss.norm.rvs(size=N, loc=1, scale=0.2, random_state=123)

        # estimate density ratio:
        n = 100
        step_size = 10e-5
        densratio = DensityRatioEstimation(n=n, epsilon=step_size)
        densratio.fit(x, y, sigma=0.1)

        # evaluate:
        test_x = np.linspace(0, 2, 11)
        test_w = ss.norm.pdf(test_x, 1, 0.1) / ss.norm.pdf(test_x, 1, 0.2)
        test_w_estim = densratio.w(test_x[:, None])

        assert np.max(np.abs(test_w - test_w_estim)) < 0.1
        assert np.abs(np.max(test_w) - densratio.max_ratio()) < 0.1

class TestCostFunction:

    def test_evaluate(self):
        def fun(x):
            return x[0]**2 + (x[1] - 1)**4
        
        cost = CostFunction(elfi.tools.vectorize(fun), None, scale=10)
        x = np.array([0.5, 0.5])
        assert np.isclose(10 * fun(x), cost.evaluate(x))

    def test_evaluate_gradient(self):
        def grad(x):
            return np.array([2 * x[0], 4 * (x[1] - 1)**3])

        cost = CostFunction(None, elfi.tools.vectorize(grad), scale=10)
        x = np.array([0.5, 0.5])
        assert np.allclose(10 * grad(x), cost.evaluate_gradient(x))
