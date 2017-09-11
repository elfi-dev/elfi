import os
import pickle

import numpy as np
import pytest

import elfi
from elfi.store import ArrayPool, ArrayStore, NpyArray, NpyStore, OutputPool


def test_npy_array():
    filename = 'test.npy'

    original = np.random.rand(3, 2)
    append = np.random.rand(10, 2)
    ones = np.ones((10, 2))
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

    # Test further appending
    arr.append(append2)
    assert np.array_equal(np.r_[original, append, append2], arr[:])
    arr.flush()
    loaded = np.load(filename)
    assert np.array_equal(np.r_[original, append, append2], loaded)

    # Test that writing over the array fails
    with pytest.raises(Exception):
        arr[len(loaded):len(loaded) + 10, :] = ones

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


def test_npy_array_multiple_instances():
    original = np.random.rand(3, 2)
    append = np.random.rand(10, 2)
    append_clone = np.random.rand(10, 2)

    filename = 'test.npy'

    # Test appending and reading
    arr = NpyArray(filename, array=original)
    arr.flush()
    arr.append(append)
    assert (len(arr) == 13)

    arr.fs.flush()

    # Make a second instance and a simultaneous append
    arr_clone = NpyArray(filename)
    arr_clone.append(append_clone)
    assert len(arr_clone) == 13
    assert np.array_equal(arr_clone[:], np.r_[original, append_clone])

    arr.close()
    arr_clone.close()

    os.remove(filename)


def test_array_pool(ma2):
    pool = ArrayPool(['MA2', 'S1'])
    N = 100
    bs = 100
    total = 1000
    rej_pool = elfi.Rejection(ma2['d'], batch_size=bs, pool=pool)
    means = rej_pool.sample(N, n_sim=total).sample_means_array

    assert len(pool.stores['MA2']) == total / bs
    assert len(pool.stores['S1']) == total / bs
    assert len(pool) == total / bs
    assert not 't1' in pool.stores

    batch2 = pool[2]

    # Test against in memory pool with using batches
    pool2 = OutputPool(['MA2', 'S1'])
    rej = elfi.Rejection(ma2['d'], batch_size=bs, pool=pool2, seed=pool.seed)
    rej.sample(N, n_sim=total)
    for bi in range(int(total / bs)):
        assert np.array_equal(pool.stores['S1'][bi], pool2.stores['S1'][bi])

    # Test running the inference again
    rej_pool.sample(N, n_sim=total)

    # Test using the same pool with another sampler
    rej_pool_new = elfi.Rejection(ma2['d'], batch_size=bs, pool=pool)
    assert len(pool) == total / bs
    assert np.array_equal(means, rej_pool_new.sample(N, n_sim=total).sample_means_array)

    # Test closing and opening the pool
    pool.close()
    pool = ArrayPool.open(pool.name)
    assert len(pool) == total / bs
    pool.close()

    # Test opening from a moved location
    os.rename(pool.path, pool.path + '_move')
    pool = ArrayPool.open(pool.name + '_move')
    assert len(pool) == total / bs
    assert np.array_equal(pool[2]['S1'], batch2['S1'])

    # Test adding a random .npy file
    r = np.random.rand(3 * bs)
    newfile = os.path.join(pool.path, 'test.npy')
    arr = NpyArray(newfile, r)
    pool.add_store('test', ArrayStore(arr, bs))
    assert len(pool.get_store('test')) == 3
    assert np.array_equal(pool[2]['test'], r[-bs:])

    # Test removing the pool
    pool.delete()
    assert not os.path.exists(pool.path)

    # Remove the pool container folder
    os.rmdir(pool.prefix)


def run_basic_store_tests(store, content):
    """

    Parameters
    ----------
    store : StoreBase
    content : nd.array

    Returns
    -------

    """
    bs = store.batch_size
    shape = content.shape[1:]
    batch = np.random.rand(bs, *shape)
    l = len(content) // bs

    assert len(store) == l

    assert np.array_equal(store[1], content[bs:2 * bs])

    store[1] = batch

    assert len(store) == l
    assert np.array_equal(store[1], batch)

    del store[l - 1]

    assert len(store) == l - 1

    store[l - 1] = batch
    assert len(store) == l

    store.clear()
    assert len(store) == 0

    # Return the original condition
    for i in range(l):
        store[i] = content[i * bs:(i + 1) * bs]

    assert len(store) == l

    return store


def test_array_store():
    arr = np.random.rand(40, 2)
    store = ArrayStore(arr, batch_size=10, n_batches=3)

    with pytest.raises(IndexError):
        store[4] = np.zeros((10, 2))

    run_basic_store_tests(store, arr[:30])


def test_npy_store():
    filename = 'test'
    arr = np.random.rand(40, 2)
    NpyArray(filename, arr).close()
    store = NpyStore(filename, batch_size=10, n_batches=4)

    run_basic_store_tests(store, arr)

    batch = np.random.rand(10, 2)
    store[4] = batch
    store[5] = 2 * batch

    assert np.array_equal(store[5], 2 * batch)

    with pytest.raises(IndexError):
        store[7] = 3 * batch

    store.delete()
