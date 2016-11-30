import elfi
import numpy as np


def test_new_inference_task():
    """This test fails if keys that the dask scheduler gets are not different. We
    run the loop 10 times in trying to get key collisions. The collisions occur
    when dask is unable to clear the key of previous computation in time before
    the next computation with the exact same key comes in. This can happen at least
    with the Distributed scheduler.

    """
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

        p = elfi.Prior('p', 'Uniform', i)
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


def test_reset_specific_scheduler_keys():
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
