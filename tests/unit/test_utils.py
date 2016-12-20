from elfi.utils import stochastic_optimization


class TestStochasticOptimization():

    def test_1dim_x2(self):
        fun = lambda x : x.dot(x)
        bounds = ((-1, 1),)
        its = int(1e3)
        polish=True
        loc, val = stochastic_optimization(fun, bounds, its, polish)
        assert abs(loc - 0.0) < 1e-5
        assert abs(val - 0.0) < 1e-5

