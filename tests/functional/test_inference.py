import pytest

from collections import OrderedDict

import numpy as np
import elfi
import examples.ma2 as ma2


def test_inference_with_informative_data():
    true_params = OrderedDict([('t1', .6), ('t2', .2)])
    n_obs = 100
    N = 1000

    # In our implementation, seed 4 gives informative (enough) synthetic observed
    # data of length 100 for quite accurate inference of the true parameters using
    # posterior mean as the point estimate
    m = ma2.get_model(n_obs=n_obs, true_params=true_params.values(), seed_obs=4)

    rej = elfi.Rejection(m['d'], batch_size=20000)
    res = rej.sample(N, quantile=.01)

    outputs = res['outputs']
    t1 = outputs['t1']
    t2 = outputs['t2']

    assert len(t1) == N

    error_bound = 0.05
    assert np.abs(np.mean(t1) - true_params['t1']) < error_bound
    assert np.abs(np.mean(t2) - true_params['t2']) < error_bound
