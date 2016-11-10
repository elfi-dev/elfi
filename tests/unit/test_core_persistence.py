import time
import timeit

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
    #print(a.key, a.dask)
    a.compute()
    td = timeit.default_timer() - t0
    assert td > sleep_time

    # Allow some time for the system to mark the simulation cached
    time.sleep(0.01)

    t0 = timeit.default_timer()
    a = sim.acquire(1)
    #print(a.key, a.dask)
    a.compute()
    td = timeit.default_timer() - t0
    assert td < sleep_time


def test_worker_memory_cache():
    sleep_time = .2
    simfn = get_sleep_simulator(sleep_time)
    sim = elfi.Simulator('sim', simfn, observed=0, store=elfi.MemoryStore())
    run_cache_test(sim, sleep_time)


def test_local_object_cache():
    sleep_time = .2
    simfn = get_sleep_simulator(sleep_time)
    local_store = np.zeros((10,1))
    sim = elfi.Simulator('sim', simfn, observed=0, store=local_store)
    run_cache_test(sim, sleep_time)
    #sim.acquire(1).compute()
    assert local_store[0][0] == 1