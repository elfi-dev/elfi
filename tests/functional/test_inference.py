import pytest

from collections import OrderedDict

import numpy as np
import elfi
import examples.ma2 as ma2


def test_rejection():
    true_params = OrderedDict(t1=.6, t2=.2)
    n_obs = 500
    N = 1000
    m = ma2.get_model(n_obs=n_obs, true_params=true_params.values())

    rej = elfi.Rejection(m.get_reference('d'), seed=23022017, batch_size=10000)
    res = rej.sample(N, quantile=.01)

    outputs = res['outputs']
    t1 = outputs['t1']
    t2 = outputs['t2']

    assert len(t1) == N
    # Set somewhat loose intervals for now
    e = 0.1
    assert np.abs(np.mean(t1) - true_params['t1']) < e
    assert np.abs(np.mean(t2) - true_params['t2']) < e
