import os

import numpy as np

import elfi
from elfi.store import OutputPool, NpyPersistedArray, ArrayPool


# TODO: npy_persisted_array rewriting of data.


def test_npy_persisted_array():
    filename = 'test.npy'

    original = np.random.rand(3, 2)

    arr = NpyPersistedArray(filename, truncate=True)
    arr.append(original)
    assert np.array_equal(original, arr[:])
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(original, loaded)

    # Test appending and reading
    arr = NpyPersistedArray(filename)
    append = np.random.rand(100, 2)
    arr.append(append)
    arr.flush()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append], loaded)

    append2 = np.random.rand(23, 2)
    arr.append(append2)
    assert np.array_equal(np.r_[original, append, append2], arr[:])
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append, append2], loaded)

    # Test truncate method
    arr = NpyPersistedArray(filename)
    arr.truncate(len(original))
    assert np.array_equal(original, arr[:])
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(original, loaded)

    # Try that truncation in initialization works
    arr = NpyPersistedArray(filename, truncate=True)
    arr.append(append)
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(append, loaded)

    os.remove(filename)


def test_array_pool(ma2):
    pool = ArrayPool(['MA2', 'S1'])
    N = 100
    bs = 100
    total = 1000
    rej_pool = elfi.Rejection(ma2['d'], batch_size=bs, pool=pool)
    rej_pool.sample(N, n_sim=total)

    assert len(pool.stores['MA2']) == total/bs
    assert len(pool.stores['S1']) == total/bs
    assert len(pool) == total/bs
    assert not 't1' in pool.stores

    # Test against in memory pool with using batches
    pool2 = OutputPool(['MA2', 'S1'])
    rej = elfi.Rejection(ma2['d'], batch_size=bs, pool=pool2, seed=pool.seed)
    rej.sample(N, n_sim=total)
    for bi in range(int(total/bs)):
        assert np.array_equal(pool.stores['S1'][bi], pool2.stores['S1'][bi])

    # Test running the inference again
    rej_pool.sample(N, n_sim=total)

    # Test using the same pool with another sampler
    rej_pool_new = elfi.Rejection(ma2['d'], batch_size=bs, pool=pool)
    assert len(pool) == total/bs

    # Test removing the pool
    pool.delete()
    assert not os.path.exists(pool.arraypath)


