import pytest
import numpy as np

import elfi
from elfi.store import DictListStore


class TestSMCDistribution():

    def get_smc(self):
        pop = [1, 5, 10]
        weights = [10, 100, 1000]
        return elfi.SMCProposal(pop, weights)

    def test_resample(self):
        smc = self.get_smc()
        rs = np.random.RandomState(123)
        s = smc.resample(10000, random_state=rs)
        p = []
        for w in smc.samples:
            p.append(np.sum(s==w)/len(s))

        assert len(np.unique(s)) == 3
        assert np.sum((p - smc.weights)**2) < 0.1

    def test_rvs_and_pdf(self):
        smc = self.get_smc()
        rs = np.random.RandomState(123)
        s_rand = rs.choice(smc.samples.ravel(), size=1000)
        rs.seed(123)
        s_weighted = rs.choice(smc.samples.ravel(), size=1000, p=smc.weights)
        rs.seed(123)
        s = smc.rvs(size=1000, random_state=rs)

        assert len(np.unique(s)) == 1000

        assert np.sum(smc.pdf(s_rand)) < np.sum(smc.pdf(s_weighted))
        assert np.sum(smc.pdf(s_rand)) < np.sum(smc.pdf(s))
        assert np.sum(smc.pdf(s)) < np.sum(smc.pdf(s_weighted))

        assert np.abs(s.mean() - np.sum(smc.weights*smc.samples)) < 10

        I = np.sum(smc.pdf(np.arange(-100, 100, .25)))*.25
        assert I > .99
        assert I < 1.01

    def test_rvs_shape(self):
        smc = self.get_smc()
        assert smc.rvs(3).shape == (3,1)

        smc = elfi.SMCProposal([[1,1], [2,2]])
        assert smc.rvs(1).shape == (1,2)


# TODO: Rewrite as a InferenceTask, do not derive subclasses from this
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


class TestBOLFI(MockModel):

    def set_basic_bolfi(self):
        # Restrict the number of workers
        elfi.env.client(n_workers=4, threads_per_worker=1)
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
