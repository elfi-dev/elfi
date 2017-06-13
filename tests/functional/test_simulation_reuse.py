import pytest
import time

import numpy as np
import scipy.stats as ss

import elfi


@pytest.mark.parametrize('sleep_model', [.2], indirect=['sleep_model'])
def test_pool(sleep_model):
    # Add nodes to the pool
    pool = elfi.OutputPool(outputs=sleep_model.parameter_names + ['slept', 'summary', 'd'])

    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    quantile = .25
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    # Will make 5/.25 = 20 evaluations with mean time of .1 secs, so 2 secs total on
    # average. Allow some slack although still on rare occasions this may fail.
    assert td > 1.3

    # Instantiating new inference with the same pool should be faster because we
    # use the prepopulated pool
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    assert td < 1.3

    # It should work if we remove the simulation, since the Rejection sampling
    # only requires the parameters and the discrepancy
    pool.remove_store('slept')
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    assert td < 1.3

    # It should work even if we remove the discrepancy, since the discrepancy can be recomputed
    # from the stored summary
    pool.remove_store('d')
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    ts = time.time()
    res = rej.sample(5, quantile=quantile)
    td = time.time() - ts
    assert td < 1.3














