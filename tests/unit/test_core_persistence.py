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
    sim.acquire(1).compute()
    td = timeit.default_timer() - t0
    assert td > sleep_time

    t0 = timeit.default_timer()
    sim.acquire(1).compute()
    td = timeit.default_timer() - t0
    assert td < sleep_time


def test_worker_memory_cache():
    sleep_time = .2
    simfn = get_sleep_simulator(sleep_time)
    sim = elfi.Simulator('sim', simfn, observed=0, store=elfi.MemoryStore())
    run_cache_test(sim, sleep_time)


def test_local_object_cache():
    sleep_time = .2
    gets = [dask.async.get_sync,
                  dask.threaded.get,
                  dask.multiprocessing.get,
                  #elfi.env.client().get
                  ]
    for g in gets:
        dask.set_options(get=g)
        simfn = get_sleep_simulator(sleep_time)
        local_store = np.zeros((10,1))
        sim = elfi.Simulator('sim', simfn, observed=0, store=local_store)
        run_cache_test(sim, sleep_time)
        assert local_store[0][0] == 1