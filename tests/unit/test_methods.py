import numpy as np
from functools import partial
import GPy
import elfi


# Tests for the base class
class Test_ABCMethod:

    def test_constructor(self):
        p1 = elfi.Prior('p1', 'uniform', 0, 1)
        p2 = elfi.Prior('p2', 'uniform', 0, 1)
        d = elfi.Discrepancy('d', np.mean, p1, p2)
        abc = elfi.ABCMethod(d, [p1, p2])

        try:
            abc = elfi.ABCMethod()
            abc = elfi.ABCMethod(0.2, None)
            abc = elfi.ABCMethod([d], [p1, p2])
            abc = elfi.ABCMethod(d, p1)
            assert False
        except:
            assert True

    def test_sample(self):
        p1 = elfi.Prior('p1', 'uniform', 0, 1)
        d = elfi.Discrepancy('d', np.mean, p1)
        abc = elfi.ABCMethod(d, [p1])
        try:
            abc.sample()
            assert False
        except:
            assert True


# Tests for rejection sampling
class Test_Rejection:

    def test_sample(self):
        p1 = elfi.Prior('p1', 'uniform', 0, 1)
        Y = elfi.Simulator('Y', lambda a, n_sim, prng: a, p1, observed=1)
        d = elfi.Discrepancy('d', lambda d1, d2: d1, Y)

        rej = elfi.Rejection(d, [p1])
        n = 200
        try:
            # some kind of test for quantile-based rejection
            result = rej.sample(n, quantile=0.5)
            assert isinstance(result, dict)
            assert 'samples' in result.keys()
            assert result['samples'][0].shape == (n, 1)
            avg = result['samples'][0].mean(axis=0)
            assert abs(avg-0.25) < 0.1
            assert abs(result['threshold']-0.5) < 0.1

            # some kind of test for threshold-based rejection
            threshold = 0.5
            result2 = rej.sample(n, threshold=threshold)
            n2 = result2['samples'][0].shape[0]
            assert n2 > n * threshold * 0.8 and n2 < n * threshold * 1.2
            assert np.all(result2['samples'][0] < threshold)
        except:
            assert False, "Possibly a random effect; try again."


class Test_BOLFI():

    def mock_simulator(self, p, prng=None):
        self.mock_sim_calls += 1
        pd = int(p*100)
        return np.atleast_1d([0] * pd + [1] * (100 - pd))

    def mock_summary(self, x):
        self.mock_sum_calls += 1
        m = np.mean(x)
        return np.atleast_1d(m)

    def mock_discrepancy(self, x, y):
        self.mock_dis_calls += 1
        d = np.linalg.norm(np.array(x).ravel() - np.array(y).ravel())
        return np.atleast_1d(d)

    def set_simple_model(self):
        self.mock_sim_calls = 0
        self.mock_sum_calls = 0
        self.mock_dis_calls = 0
        self.bounds = ((0, 1),)
        self.input_dim = 1
        self.obs = self.mock_simulator(0.5)
        self.mock_sim_calls = 0
        self.p = elfi.Prior('p', 'uniform', 0, 1)
        self.Y = elfi.Simulator('Y', self.mock_simulator, self.p, observed=self.obs, vectorized=False)
        self.S = elfi.Summary('S', self.mock_summary, self.Y)
        self.d = elfi.Discrepancy('d', self.mock_discrepancy, self.S)

    def set_basic_bolfi(self):
        self.n_sim = 2
        self.n_batch = 1
        self.kernel_class = "Matern32"
        self.kernel_var = 1.0
        self.kernel_scale = 1.0
        self.model = elfi.GPyModel(input_dim=self.input_dim,
                              bounds=self.bounds,
                              kernel_class=self.kernel_class,
                              kernel_var=self.kernel_var,
                              kernel_scale=self.kernel_scale)
        self.acq = elfi.BolfiAcquisition(self.model, n_samples=self.n_sim,
                                    exploration_rate=2.5, opt_iterations=1000)

    def test_basic_sync_use(self):
        self.set_simple_model()
        self.set_basic_bolfi()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           model=self.model, acquisition=self.acq, sync=True)
        post = bolfi.infer()
        assert self.acq.finished is True

    def test_basic_async_use(self):
        self.set_simple_model()
        self.set_basic_bolfi()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           model=self.model, acquisition=self.acq, sync=False)
        post = bolfi.infer()
        assert self.acq.finished is True
