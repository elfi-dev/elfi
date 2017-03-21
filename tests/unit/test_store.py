import pytest
import os

import numpy as np
import scipy.stats as ss

import elfi
from elfi.store import OutputPool, NpyPersistedArray, ArrayPool


def test_npy_persisted_array():
    filename = 'test.npy'

    original = np.random.rand(3, 2)

    arr = NpyPersistedArray(filename, mode='w')
    arr.append(original)
    read = arr[:]
    assert np.array_equal(original, read)
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(original, loaded)

    # Test appending and reading
    arr = NpyPersistedArray(filename, mode='a')
    append = np.random.rand(100, 2)
    arr.append(append)
    arr.flush()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append], loaded)

    append2 = np.random.rand(23, 2)
    arr.append(append2)
    read = arr[:]
    assert np.array_equal(np.r_[original, append, append2], read)
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append, append2], loaded)

    # Try that truncation works
    arr = NpyPersistedArray(filename, mode='w')
    arr.append(append)
    arr.close()
    loaded = np.load(filename)
    assert np.array_equal(append, loaded)

    os.remove(filename)


def test_array_pool(ma2):
    pool = ArrayPool(['MA2', 'S1'])
    N = 100
    p = .1
    rej = elfi.Rejection(ma2['d'], batch_size=100, pool=pool)
    rej.sample(N, p=p)

    assert len(pool['MA2']) == N/p
    assert len(pool['S1']) == N/p
    assert not 't1' in pool

    # Test against otherwise generated pool
    pool2 = OutputPool(['MA2', 'S1'])
    rej = elfi.Rejection(ma2['d'], batch_size=100, pool=pool2, seed=pool.seed)
    rej.sample(N, p=p)

    for bi in range(int(1/p)):
        assert np.array_equal(pool['S1'][bi], pool2['S1'][bi])

    pool.destroy()

    assert not os.path.exists(pool.path)
