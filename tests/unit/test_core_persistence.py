import numpy as np
import time
import timeit

import elfi

from elfi.core import LocalDataStore

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


# TODO: is this needed any longer?
def clear_elfi_client():
    elfi.env.client().shutdown()
    elfi.env.set(client=None)


class TestPersistence():

    def test_worker_memory_cache(self):
        sleep_time = .2
        simfn = get_sleep_simulator(sleep_time)
        sim = elfi.Simulator("sim", simfn, observed=0, store=elfi.MemoryStore())
        res = run_cache_test(sim, sleep_time)
        assert res[0][0] == 1

        # Test that nodes derived from `sim` benefit from the caching
        summ = elfi.Summary("sum", lambda x: x, sim)
        t0 = timeit.default_timer()
        res = summ.acquire(1).compute()
        td = timeit.default_timer() - t0
        assert td < sleep_time
        assert res[0][0] == 1

        elfi.env.client().shutdown()

    def test_local_object_cache(self):
        local_obj = np.zeros((10,1))
        local_store = LocalDataStore(local_obj)
        self.run_local_object_cache_test(local_store)

    def run_local_object_cache_test(self, local_store):
        sleep_time = .2
        simfn = get_sleep_simulator(sleep_time)
        sim = elfi.Simulator("sim", simfn, observed=0, store=local_store)
        run_cache_test(sim, sleep_time)
        assert local_store._read_data("sim", 0)[0] == 1

        # Test that nodes derived from `sim` benefit from the storing
        summ = elfi.Summary("sum", lambda x : x, sim)
        t0 = timeit.default_timer()
        res = summ.acquire(1).compute()
        td = timeit.default_timer() - t0
        assert td < sleep_time
        assert res[0][0] == 1

        elfi.env.client().shutdown()


def test_independent_keys():
    elfi.env.client(n_workers=2, threads_per_worker=1)
    N = 20
    bs = 10

    for i in range(10):
        p1 = elfi.Prior('p', 'Uniform')
        sim1 = elfi.Simulator('sim', lambda *args, **kwargs: args[0], p1, observed=1)
        p2 = elfi.Prior('p', 'Uniform')
        sim2 = elfi.Simulator('sim', lambda *args, **kwargs: args[0], p2, observed=1)

        y1 = sim1.acquire(N, batch_size=bs).compute()
        t1 = p1.acquire(N, batch_size=bs).compute()
        y2 = sim2.acquire(N, batch_size=bs).compute()
        t2 = p2.acquire(N, batch_size=bs).compute()
        assert np.all(y1 != y2)
        assert np.all(t1 != t2)

    elfi.env.client().shutdown()