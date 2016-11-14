import time
import timeit
import pytest

import numpy as np
import dask
import distributed
import elfi


def get_sleep_simulator(sleep_time=.1, *args, **kwargs):
    def sim(*args, **kwargs):
        time.sleep(sleep_time)
        return np.array([[1]])
    return sim


def run_cache_test(sim, sleep_time):
    t0 = timeit.default_timer()
    a = sim.acquire(1)
    a.compute()
    td = timeit.default_timer() - t0
    assert td > sleep_time

    t0 = timeit.default_timer()
    a = sim.acquire(1)
    res = a.compute()
    td = timeit.default_timer() - t0
    assert td < sleep_time

    return res


def test_worker_memory_cache():
    sleep_time = .2
    simfn = get_sleep_simulator(sleep_time)
    sim = elfi.Simulator('sim', simfn, observed=0, store=elfi.MemoryStore())
    res = run_cache_test(sim, sleep_time)
    assert res[0][0] == 1

    # Test that nodes derived from `sim` benefit from the caching
    sum = elfi.Summary('sum', lambda x: x, sim)
    t0 = timeit.default_timer()
    res = sum.acquire(1).compute()
    td = timeit.default_timer() - t0
    assert td < sleep_time
    assert res[0][0] == 1

    # Restart client for later tests
    elfi.env.client().restart()


def test_local_object_cache():
    sleep_time = .2
    simfn = get_sleep_simulator(sleep_time)
    local_store = np.zeros((10,1))
    sim = elfi.Simulator('sim', simfn, observed=0, store=local_store)
    run_cache_test(sim, sleep_time)
    assert local_store[0][0] == 1

    # Test that nodes derived from `sim` benefit from the storing
    sum = elfi.Summary('sum', lambda x : x, sim)
    t0 = timeit.default_timer()
    res = sum.acquire(1).compute()
    td = timeit.default_timer() - t0
    assert td < sleep_time
    assert res[0][0] == 1

    # Restart client for later tests
    elfi.env.client().restart()