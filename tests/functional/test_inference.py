import pytest
import logging
import sys

from collections import OrderedDict

import numpy as np
import elfi
import examples.ma2 as ma2


def setup_ma2_with_informative_data():
    true_params = OrderedDict([('t1', .6), ('t2', .2)])
    n_obs = 100

    # In our implementation, seed 4 gives informative (enough) synthetic observed
    # data of length 100 for quite accurate inference of the true parameters using
    # posterior mean as the point estimate
    m = ma2.get_model(n_obs=n_obs, true_params=true_params.values(), seed_obs=4)
    return m, true_params


def check_inference_with_informative_data(res, N, true_params, error_bound=0.05):
    outputs = res['outputs']
    t1 = outputs['t1']
    t2 = outputs['t2']

    assert len(t1) == N

    assert np.abs(np.mean(t1) - true_params['t1']) < error_bound, \
        "\n\nNot |{} - {}| < {}\n".format(np.mean(t1), true_params['t1'], error_bound)
    assert np.abs(np.mean(t2) - true_params['t2']) < error_bound, \
        "\n\nNot |{} - {}| < {}\n".format(np.mean(t2), true_params['t2'], error_bound)


@pytest.mark.usefixtures('with_all_clients')
def test_rejection_with_quantile():
    m, true_params = setup_ma2_with_informative_data()

    q = 0.01
    N = 1000
    batch_size = 20000
    rej = elfi.Rejection(m['d'], batch_size=batch_size)
    res = rej.sample(N, quantile=q)

    check_inference_with_informative_data(res, N, true_params)

    # Check that there are no repeating values indicating a seeding problem
    assert len(np.unique(res['outputs']['d'])) == N

    assert res['accept_rate'] == q


@pytest.mark.usefixtures('with_all_clients')
def test_rejection_with_threshold():
    m, true_params = setup_ma2_with_informative_data()

    t = .1
    N = 1000
    rej = elfi.Rejection(m['d'], batch_size=20000)
    res = rej.sample(N, threshold=t)

    check_inference_with_informative_data(res, N, true_params)

    assert res['threshold'] <= t


@pytest.mark.usefixtures('with_all_clients')
def test_bolfi():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('elfi.executor').setLevel(logging.WARNING)
    m, true_params = setup_ma2_with_informative_data()
    bolfi = elfi.BOLFI(m['d'],
                       max_parallel_acquisitions=30,
                       n_total_evidence=150,
                       initial_evidence=30,
                       update_interval=30)
    post = bolfi.infer(threshold=.01)

    # TODO: sampling to get the mean
    res = dict(outputs=dict(t1=np.array([post.ML[0]]), t2=np.array([post.ML[1]])))
    check_inference_with_informative_data(res, 1, true_params, error_bound=.1)


















