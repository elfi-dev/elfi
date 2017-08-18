import os
import time

import numpy as np
import pytest

import elfi


@pytest.mark.parametrize('sleep_model', [.2], indirect=['sleep_model'])
def test_pool_usage(sleep_model):
    # Add nodes to the pool
    pool = elfi.OutputPool(outputs=sleep_model.parameter_names + ['slept', 'summary', 'd'])

    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    quantile = .25
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    # Will make 5/.25 = 20 evaluations with mean time of .1 secs, so 2 secs total on
    # average. Allow some slack although still on rare occasions this may fail.
    assert td > 1.2

    # Instantiating new inference with the same pool should be faster because we
    # use the prepopulated pool
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    assert td < 1.2

    # It should work if we remove the simulation, since the Rejection sampling
    # only requires the parameters and the discrepancy
    pool.remove_store('slept')
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    assert td < 1.2

    # It should work even if we remove the discrepancy, since the discrepancy can be recomputed
    # from the stored summary
    pool.remove_store('d')
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    assert td < 1.2


def test_pool_restarts(ma2):
    pool = elfi.ArrayPool(['t1', 'd'], name='test')
    rej = elfi.Rejection(ma2, 'd', batch_size=10, pool=pool, seed=123)

    rej.sample(1, n_sim=30)
    pool.save()

    # Do not save the pool...
    rej = elfi.Rejection(ma2, 'd', batch_size=10, pool=pool)
    rej.set_objective(3, n_sim=60)
    while not rej.finished:
        rej.iterate()
    # ...but just flush the array content
    pool.get_store('t1').array.fs.flush()
    pool.get_store('d').array.fs.flush()

    assert (len(pool) == 6)
    assert (len(pool.stores['t1'].array) == 60)

    pool2 = elfi.ArrayPool.open('test')
    assert (len(pool2) == 3)
    assert (len(pool2.stores['t1'].array) == 30)

    rej = elfi.Rejection(ma2, 'd', batch_size=10, pool=pool2)
    s9pool = rej.sample(3, n_sim=90)
    pool2.save()

    pool2 = elfi.ArrayPool.open('test')
    rej = elfi.Rejection(ma2, 'd', batch_size=10, pool=pool2)
    s9pool_loaded = rej.sample(3, n_sim=90)

    rej = elfi.Rejection(ma2, 'd', batch_size=10, seed=123)
    s9 = rej.sample(3, n_sim=90)

    assert np.array_equal(s9pool.samples['t1'], s9.samples['t1'])
    assert np.array_equal(s9pool.discrepancies, s9.discrepancies)

    assert np.array_equal(s9pool.samples['t1'], s9pool_loaded.samples['t1'])
    assert np.array_equal(s9pool.discrepancies, s9pool_loaded.discrepancies)

    pool.delete()
    pool2.delete()

    os.rmdir(pool.prefix)
