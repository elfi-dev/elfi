import time
import timeit

import numpy as np
import elfi


def sleep_simulator(sleep = .1, *args, **kwargs):
    time.sleep(sleep)
    return np.array([0])


def test_worker_memory_cache():
    sleep_time = .1
    simfn = lambda *args, **kwargs : sleep_simulator(sleep_time, *args, **kwargs)

    sim = elfi.Simulator('sim', simfn, observed=np.array([0]), store=elfi.MemoryStore())

    t0 = timeit.default_timer()
    sim.acquire(1).compute()
    td = timeit.default_timer() - t0
    assert td > sleep_time

    t0 = timeit.default_timer()
    sim.acquire(1).compute()
    td = timeit.default_timer() - t0
    assert td < sleep_time