import numpy as np

import elfi
from elfi.utils import stochastic_optimization, weighted_cov


class TestStochasticOptimization():

    def test_1dim_x2(self):
        fun = lambda x : x.dot(x)
        bounds = ((-1, 1),)
        its = int(1e3)
        polish=True
        loc, val = stochastic_optimization(fun, bounds, its, polish)
        assert abs(loc - 0.0) < 1e-5
        assert abs(val - 0.0) < 1e-5


def test_weighted_cov():
    cov = [[.5, -.3], [-.3, .7]]
    x = np.random.RandomState(12345).multivariate_normal([1,2], cov, 1000)
    w = [1]*len(x)
    assert np.linalg.norm(weighted_cov(x, w) - cov) < .1
