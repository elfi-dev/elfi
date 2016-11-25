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


# TODO: is this needed any longer (env.py should handle clients that are shut down)?
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
        assert local_store._read_data(sim.id, 0)[0] == 1

        # Test that nodes derived from `sim` benefit from the storing
        summ = elfi.Summary("sum", lambda x : x, sim)
        t0 = timeit.default_timer()
        res = summ.acquire(1).compute()
        td = timeit.default_timer() - t0
        assert td < sleep_time
        assert res[0][0] == 1

        elfi.env.client().shutdown()


def test_new_inference_task():
    """This test fails if keys that the dask scheduler gets are not different. We
    run the loop 10 times in trying to get key collisions."""
    elfi.env.client(n_workers=2, threads_per_worker=1)
    N = 20
    bs = 10

    p_id = None
    sim_id = None
    y = None
    t = None

    for i in range(10):
        p_prev_id = p_id
        sim_prev_id = sim_id
        y_prev = y
        t_prev = t

        p = elfi.Prior('p', 'Uniform')
        sim = elfi.Simulator('sim', lambda *args, **kwargs: args[0], p, observed=1)
        y = sim.acquire(N, batch_size=bs).compute()
        t = p.acquire(N, batch_size=bs).compute()

        if y_prev is not None:
            assert np.all(y != y_prev)
            assert np.all(t != t_prev)

        p_id = p.id
        sim_id = sim.id
        if p_prev_id is not None:
            assert p_id != p_prev_id
            assert sim_id != sim_prev_id

        elfi.new_inference_task()

    elfi.env.client().shutdown()


def test_reset_specific_schduler_keys():
    """This test fails if keys are not different"""
    elfi.env.client(n_workers=2, threads_per_worker=1)
    N = 20
    bs = 10

    y = None
    t = None

    p1 = elfi.Prior('p', 'Uniform')
    sim1 = elfi.Simulator('sim', lambda *args, **kwargs: args[0], p1, observed=1)

    for i in range(10):
        y_prev = y
        t_prev = t

        y = sim1.acquire(N, batch_size=bs).compute()
        t = p1.acquire(N, batch_size=bs).compute()

        if y_prev is not None:
            assert np.all(y != y_prev)
            assert np.all(t != t_prev)

        p1.reset()

    elfi.env.client().shutdown()