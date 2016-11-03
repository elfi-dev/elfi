import elfi
import numpy as np

def test_node_data_sub_slicing():
    mu = elfi.Prior('mu', 'uniform', 0, 4)
    ar1 = mu.acquire(10).compute()
    ar2 = mu.acquire(5).compute()
    assert np.array_equal(ar1[0:5], ar2)

    ar3 = mu.acquire(20).compute()
    assert np.array_equal(ar1, ar3[0:10])

def test_generate_vs_acquire():
    mu = elfi.Prior('mu', 'uniform', 0, 4)
    ar1 = mu.acquire(10).compute()
    ar2 = mu.generate(5).compute()
    ar12 = mu.acquire(15).compute()
    assert np.array_equal(np.vstack((ar1, ar2)), ar12)

