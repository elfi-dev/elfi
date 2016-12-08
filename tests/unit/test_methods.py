import pytest

import numpy as np

import elfi
from elfi.storage import DictListStore


# Test case
class MockModel():

    def mock_simulator(self, p, batch_size=1, random_state=None):
        self.mock_sim_calls += np.atleast_2d(p).shape[0]
        return np.hstack([p, p])

    def mock_summary(self, x):
        self.mock_sum_calls += x.shape[0]
        m = np.mean(x, axis=1, keepdims=True)
        return m

    def mock_discrepancy(self, x, y):
        self.mock_dis_calls += x[0].shape[0]
        d = np.linalg.norm(np.array(x) - np.array(y), axis=0, ord=1)
        return d

    def set_simple_model(self):
        self.mock_sim_calls = 0
        self.mock_sum_calls = 0
        self.mock_dis_calls = 0
        self.bounds = ((0, 1),)
        self.input_dim = 1
        self.obs = self.mock_simulator(0.)
        self.mock_sim_calls = 0
        self.p = elfi.Prior('p', 'uniform', 0, 1)
        self.Y = elfi.Simulator('Y', self.mock_simulator, self.p,
                                observed=self.obs)
        self.S = elfi.Summary('S', self.mock_summary, self.Y)
        self.d = elfi.Discrepancy('d', self.mock_discrepancy, self.S)


# Tests for the base class
class TestABCMethod(MockModel):

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
            abc.sample()  # NotImplementedError
            assert False
        except:
            assert True

    def test_get_distances(self):
        self.set_simple_model()
        abc = elfi.ABCMethod(self.d, [self.p], batch_size=1)
        n_sim = 4
        distances, parameters = abc._acquire(n_sim)
        assert distances.shape == (n_sim, 1)
        assert isinstance(parameters, list)
        assert parameters[0].shape == (n_sim, 1)


# Tests for rejection sampling
class TestRejection(MockModel):

    def test_quantile(self):
        self.set_simple_model()

        n = 20
        batch_size = 10
        quantile = 0.5
        rej = elfi.Rejection(self.d, [self.p], batch_size=batch_size)

        result = rej.sample(n, quantile=quantile)
        assert isinstance(result, dict)
        assert 'samples' in result.keys()
        assert result['samples'][0].shape == (n, 1)
        assert self.mock_sim_calls == int(n / quantile)
        assert self.mock_sum_calls == int(n / quantile) + 1
        assert self.mock_dis_calls == int(n / quantile)

    def test_threshold(self):
        self.set_simple_model()

        n = 10
        batch_size = 10
        rej = elfi.Rejection(self.d, [self.p], batch_size=batch_size)
        threshold = 0.5

        result = rej.sample(n, threshold=threshold)
        assert isinstance(result, dict)
        assert 'samples' in result.keys()
        assert self.mock_sim_calls >= int(n)
        assert self.mock_sim_calls % batch_size == 0  # should be a multiple of batch_size for this test
        assert self.mock_sum_calls >= int(n) + 1
        assert self.mock_dis_calls >= int(n)
        assert np.all(result['samples'][0] < threshold)  # makes sense only for MockModel!

    def test_reject(self):
        self.set_simple_model()

        n = 20
        batch_size = 10
        quantile = 0.5
        rej = elfi.Rejection(self.d, [self.p], batch_size=batch_size)
        threshold = 0.1

        rej.sample(n, quantile=quantile)
        result = rej.reject(threshold=threshold)
        assert isinstance(result, dict)
        assert 'samples' in result.keys()
        assert np.all(result['samples'][0] < threshold)  # makes sense only for MockModel!


@pytest.mark.skip(reason="SMC implementation needs to be fixed")
class TestSMC(MockModel):

    def test_SMC_dist(self):
        current_params = np.array([[1.], [10.], [100.], [1000.]])
        weighted_sd = np.array([1.])
        weights = np.array([[0.], [0.], [1.], [0.]])
        weights /= np.sum(weights)
        random_state = np.random.RandomState(0)
        params = elfi.SMC_Distribution.rvs(current_params, weighted_sd, weights, random_state, size=current_params.shape)
        assert params.shape == (4, 1)
        assert np.allclose(params, current_params[2, 0], atol=5.)
        p = elfi.SMC_Distribution.pdf(params, current_params, weighted_sd, weights)
        assert p.shape == (4, 1)

    def test_SMC(self):
        self.set_simple_model()

        n = 20
        batch_size = 10
        smc = elfi.SMC(self.d, [self.p], batch_size=batch_size)
        n_populations = 3
        schedule = [0.5] * n_populations

        prior_id = id(self.p)
        result = smc.sample(n, n_populations, schedule)

        assert id(self.p) == prior_id  # changed within SMC, finally reverted
        assert self.mock_sim_calls == int(n / schedule[0] * n_populations)
        assert self.mock_sum_calls == int(n / schedule[0] * n_populations) + 1
        assert self.mock_dis_calls == int(n / schedule[0] * n_populations)


@pytest.mark.skip(reason="The Simulator must be separated from the TestBOLFI class")
class TestBOLFI(MockModel):

    def set_basic_bolfi(self):
        self.n_sim = 4
        self.n_batch = 2

    def test_basic_sync_use(self):
        self.set_simple_model()
        self.set_basic_bolfi()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           n_surrogate_samples=self.n_sim,
                           sync=True)
        post = bolfi.infer()
        assert bolfi.acquisition.finished is True
        assert bolfi.model.n_observations == self.n_sim

    def test_basic_async_use(self):
        self.set_simple_model()
        self.set_basic_bolfi()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           n_surrogate_samples=self.n_sim,
                           sync=False)
        post = bolfi.infer()
        assert bolfi.acquisition.finished is True
        assert bolfi.model.n_observations == self.n_sim

    def test_optimization(self):
        self.set_simple_model()
        self.set_basic_bolfi()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           n_surrogate_samples=self.n_sim,
                           n_opt_iters=10)
        post = bolfi.infer()
        assert bolfi.acquisition.finished is True
        assert bolfi.model.n_observations == self.n_sim

    def test_model_logging(self):
        self.set_simple_model()
        self.set_basic_bolfi()
        db = DictListStore()
        bolfi = elfi.BOLFI(self.d, [self.p], self.n_batch,
                           store=db,
                           n_surrogate_samples=self.n_sim,
                           n_opt_iters=10)
        post = bolfi.infer()
        assert bolfi.acquisition.finished is True
        assert bolfi.model.n_observations == self.n_sim
        # get initial model plus resulting model after each sample
        models = db.get("BOLFI-model", slice(0, self.n_sim+1))
        assert len(models) == self.n_sim + 1
        for i in range(self.n_sim+1):
            assert type(models[i]) == type(bolfi.model), i

