import numpy as np
from functools import partial
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
