import pytest

import numpy as np
import elfi
import elfi.examples.ma2 as ma2


def test_rejection():
    elfi.env.client(4,1)
    t1_0 = .6
    t2_0 = .2
    N = 1000
    itask = ma2.inference_task(500, true_params=[t1_0, t2_0])
    rej = elfi.Rejection(itask.discrepancy, itask.parameters, batch_size=20000)
    res = rej.sample(N, quantile=.01)
    samples = res["samples"]

    assert isinstance(samples, list)
    assert len(samples[0]) == N
    # Set somewhat loose intervals for now
    e = 0.12
    assert np.abs(np.mean(samples[0]) - t1_0) < e
    assert np.abs(np.mean(samples[1]) - t2_0) < e

    elfi.env.client().shutdown()


def test_smc():
    elfi.env.client(4,1)
    t1_0 = .6
    t2_0 = .2
    N = 1000
    itask = ma2.inference_task(500, true_params=[t1_0, t2_0])

    smc = elfi.SMC(itask.discrepancy, itask.parameters, batch_size=50000)
    res = smc.sample(N, 3, schedule=[1, 0.5, 0.1])

    assert not np.array_equal(res["samples_history"][0][0], res["samples_history"][1][0])
    assert not np.array_equal(res["samples_history"][-1][0], res["samples"][0])
    assert not np.array_equal(res["weights_history"][0], res["weights_history"][1])
    assert not np.array_equal(res["weights"][-1], res["weights"])
    assert not np.array_equal(res["weights_history"][0], res["weights_history"][1])
    assert not np.array_equal(res["weights"][-1], res["weights"])

    s = res['samples']
    w = res['weights']
    assert len(s[0]) == N

    # Set somewhat loose intervals for now
    e = 0.12
    assert np.abs(np.average(s[0], axis=0, weights=w) - t1_0) < e
    assert np.abs(np.average(s[1], axis=0, weights=w) - t2_0) < e

    elfi.env.client().shutdown()


