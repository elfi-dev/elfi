import pytest
import logging
import time
import sys

from collections import OrderedDict

import numpy as np
import elfi
from elfi.model.elfi_model import NodeReference
import examples.ma2 as ma2


slow = pytest.mark.skipif(
    pytest.config.getoption("--skipslow"),
    reason="--skipslow argument given"
)


def setup_ma2_with_informative_data():
    true_params = OrderedDict([('t1', .6), ('t2', .2)])
    n_obs = 100

    # In our implementation, seed 4 gives informative (enough) synthetic observed
    # data of length 100 for quite accurate inference of the true parameters using
    # posterior mean as the point estimate
    m = ma2.get_model(n_obs=n_obs, true_params=true_params.values(), seed_obs=4)
    return m, true_params


def check_inference_with_informative_data(res, N, true_params, error_bound=0.05):
    outputs = res['samples']
    t1 = outputs['t1']
    t2 = outputs['t2']

    if N > 1:
        assert len(t1) == N

    assert np.abs(np.mean(t1) - true_params['t1']) < error_bound, \
        "\n\nNot |{} - {}| < {}\n".format(np.mean(t1), true_params['t1'], error_bound)
    assert np.abs(np.mean(t2) - true_params['t2']) < error_bound, \
        "\n\nNot |{} - {}| < {}\n".format(np.mean(t2), true_params['t2'], error_bound)


@pytest.mark.usefixtures('with_all_clients')
def test_rejection_with_quantile():
    m, true_params = setup_ma2_with_informative_data()

    p = 0.01
    N = 1000
    batch_size = 20000
    rej = elfi.Rejection(m['d'], batch_size=batch_size)
    res = rej.sample(N, p=p)

    check_inference_with_informative_data(res, N, true_params)

    # Check that there are no repeating values indicating a seeding problem
    assert len(np.unique(res['samples']['d'])) == N

    assert res['accept_rate'] == p
    assert res['n_sim'] == int(N/p)


@pytest.mark.usefixtures('with_all_clients')
def test_rejection_with_threshold():
    m, true_params = setup_ma2_with_informative_data()

    t = .1
    N = 1000
    rej = elfi.Rejection(m['d'], batch_size=20000)
    res = rej.sample(N, threshold=t)

    check_inference_with_informative_data(res, N, true_params)

    assert res['threshold'] <= t
    # Test that we got unique samples (no repeating of batches).
    assert len(np.unique(res['samples']['d'])) == N


@pytest.mark.usefixtures('with_all_clients')
def test_smc():
    m, true_params = setup_ma2_with_informative_data()

    thresholds = [.5, .25, .1]
    N = 1000
    smc = elfi.SMC(m['d'], batch_size=20000)
    res = smc.sample(N, thresholds=thresholds)

    check_inference_with_informative_data(res, N, true_params)

    # We should be able to carry out the inference in less than six batches
    assert res['n_batches'] < 6

@slow
@pytest.mark.usefixtures('with_all_clients')
def test_bayesian_optimization():
    m, true_params = setup_ma2_with_informative_data()

    # Log distance tends to work better
    log_d = NodeReference(m['d'], state=dict(_operation=np.log), model=m, name='log_d')

    bo = elfi.BayesianOptimization(log_d,
                                   n_acq=300,
                                   batch_size=5,
                                   initial_evidence=20,
                                   update_interval=10,
                                   bounds=[(-2,2)]*len(m.parameters))
    res = bo.infer()

    assert bo.target_model.n_evidence == 320
    acq_x = bo.target_model._gp.X
    check_inference_with_informative_data(res, 1, true_params, error_bound=.2)

    # Test that you can continue the inference where we left off
    res = bo.infer(n_acq=310)
    assert bo.target_model.n_evidence == 330
    assert np.array_equal(bo.target_model._gp.X[:320,:], acq_x)

@pytest.mark.skip
@slow
@pytest.mark.usefixtures('with_all_clients')
def test_BOLFI():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('elfi.executor').setLevel(logging.WARNING)

    m, true_params = setup_ma2_with_informative_data()
    bo = elfi.BayesianOptimization(m['d'], initial_evidence=30, update_interval=30)
    post = bo.infer(threshold=.01)

    # TODO: sampling to get the mean
    res = dict(outputs=dict(t1=np.array([post.ML[0]]), t2=np.array([post.ML[1]])))
    check_inference_with_informative_data(res, 1, true_params, error_bound=.1)


@pytest.mark.parametrize('sleep_model', [.2], indirect=['sleep_model'])
def test_pool(sleep_model):
    # Add nodes to the pool
    pool = elfi.OutputPool(outputs=sleep_model.parameters + ['slept', 'd'])
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)

    p = .25
    ts = time.time()
    res = rej.sample(5, p=p)
    td = time.time() - ts

    # Will make 5/.25 = 20 evaluations with mean time of .1 secs, so 2 secs total on
    # average. Allow some slack although still on rare occasions this may fail.
    assert td > 1.3

    # Instantiating new inference with the same pool should be faster because we
    # use the prepopulated pool
    rej = elfi.Rejection(sleep_model['d'], batch_size=5, pool=pool)
    ts = time.time()
    res = rej.sample(5, p=p)
    td = time.time() - ts

    assert td < 1.3

    print(res)

















