from elfi.utils import stochastic_optimization, weighted_var
import numpy as np


class Test_stochastic_optimization():

    def test_1dim_x2(self):
        fun = lambda x : x.dot(x)
        bounds = ((-1, 1),)
        its = int(1e3)
        polish=True
        loc, val = stochastic_optimization(fun, bounds, its, polish)
        assert abs(loc - 0.0) < 1e-5
        assert abs(val - 0.0) < 1e-5


class Test_weighted_var():

    def test_weighted_var(self):
        data = np.random.randn(10, 4)
        weights = np.ones(10)
        wvar = weighted_var(data, weights)
        assert wvar.shape == (4,)
        assert weights.shape == (10,)
        assert data.shape == (10, 4)
        assert np.allclose(wvar, np.var(data, axis=0))
        weights[:3] = 0.
        wvar = weighted_var(data, weights)
        assert np.allclose(wvar, np.var(data[3:,:], axis=0))
