import os
import pickle

import numpy as np

import elfi
from elfi.store import OutputPool, NpyArray, ArrayPool


def test_npy_persisted_array():
    filename = 'test.npy'

    original = np.random.rand(3, 2)
    append = np.random.rand(10, 2)
    ones = np.ones((10,2))
    append2 = np.random.rand(23, 2)

    arr = NpyArray(filename, truncate=True)
    arr.append(original)
    assert np.array_equal(original, arr[:])
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(original, loaded)

    # Test appending and reading
    arr = NpyArray(filename)
    arr.append(append)
    arr.flush()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append], loaded)

    arr.append(append2)
    assert np.array_equal(np.r_[original, append, append2], arr[:])
    arr.flush()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append, append2], loaded)

    # Test rewriting
    arr[3:13, :] = ones
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, ones, append2], loaded)

    # Test pickling
    arr = NpyArray(filename)
    serialized = pickle.dumps(arr)
    arr = pickle.loads(serialized)
    assert np.array_equal(np.r_[original, ones, append2], arr[:])

    # Test truncate method
    arr = NpyArray(filename)
    arr.truncate(len(original))
    assert np.array_equal(original, arr[:])
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(original, loaded)

    # Try that truncation in initialization works
    arr = NpyArray(filename, truncate=True)
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
    means = rej_pool.sample(N, n_sim=total).sample_means_array

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
    assert np.array_equal(means, rej_pool_new.sample(N, n_sim=total).sample_means_array)

    # Test closing and opening the pool
    pool.close()
    pool = ArrayPool.open(pool.name)
    assert len(pool) == total/bs

    # Test removing the pool
    pool.delete()
    assert not os.path.exists(pool.path)

    # Remove the pool container folder
    os.rmdir(pool.prefix)



