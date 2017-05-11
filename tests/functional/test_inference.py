import pytest
import logging

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


def check_inference_with_informative_data(outputs, N, true_params, error_bound=0.05):
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

    quantile = 0.01
    N = 1000
    batch_size = 20000
    rej = elfi.Rejection(m['d'], batch_size=batch_size)
    res = rej.sample(N, quantile=quantile)

    check_inference_with_informative_data(res.samples, N, true_params)

    # Check that there are no repeating values indicating a seeding problem
    assert len(np.unique(res.discrepancy)) == N

    assert res.accept_rate == quantile
    assert res.n_sim == int(N/quantile)


@pytest.mark.usefixtures('with_all_clients')
def test_rejection_with_threshold():
    m, true_params = setup_ma2_with_informative_data()

    t = .1
    N = 1000
    rej = elfi.Rejection(m['d'], batch_size=20000)
    res = rej.sample(N, threshold=t)

    check_inference_with_informative_data(res.samples, N, true_params)

    assert res.threshold <= t
    # Test that we got unique samples (no repeating of batches).
    assert len(np.unique(res.discrepancy)) == N


@pytest.mark.usefixtures('with_all_clients')
def test_smc():
    m, true_params = setup_ma2_with_informative_data()

    thresholds = [.5, .25, .1]
    N = 1000
    smc = elfi.SMC(m['d'], batch_size=20000)
    res = smc.sample(N, thresholds=thresholds)

    check_inference_with_informative_data(res.samples, N, true_params)

    # We should be able to carry out the inference in less than six batches
    assert res.populations[-1].n_batches < 6


@pytest.mark.usefixtures('with_all_clients')
def test_BO():
    m, true_params = setup_ma2_with_informative_data()

    # Log discrepancy tends to work better
    log_d = NodeReference(m['d'], state=dict(_operation=np.log), model=m, name='log_d')

    n_init = 20
    res_init = elfi.Rejection(log_d, batch_size=5).sample(n_init, quantile=1)

    bo = elfi.BayesianOptimization(log_d, initial_evidence=res_init.outputs, update_interval=10, batch_size=5,
                       bounds=[(-2,2)]*len(m.parameters))
    assert bo.target_model.n_evidence == n_init
    assert bo.n_evidence == n_init
    assert bo._n_precomputed == n_init
    assert bo.n_initial_evidence == n_init

    n1 = 5
    res = bo.infer(n_init + n1)

    assert bo.target_model.n_evidence == n_init + n1
    assert bo.n_evidence == n_init + n1
    assert bo._n_precomputed == n_init
    assert bo.n_initial_evidence == n_init

    n2 = 5
    res = bo.infer(n_init + n1 + n2)

    assert bo.target_model.n_evidence == n_init + n1 + n2
    assert bo.n_evidence == n_init + n1 + n2
    assert bo._n_precomputed == n_init
    assert bo.n_initial_evidence == n_init

    assert np.array_equal(bo.target_model._gp.X[:n_init, 0], res_init.samples_list[0])


@slow
@pytest.mark.usefixtures('with_all_clients')
def test_BOLFI():
    # logging.basicConfig(level=logging.DEBUG)
    # logging.getLogger('elfi.executor').setLevel(logging.WARNING)

    m, true_params = setup_ma2_with_informative_data()

    # Log discrepancy tends to work better
    log_d = NodeReference(m['d'], state=dict(_operation=np.log), model=m, name='log_d')

    bolfi = elfi.BOLFI(log_d, initial_evidence=20, update_interval=10, batch_size=5,
                       bounds=[(-2,2)]*len(m.parameters))
    res = bolfi.infer(300)
    assert bolfi.target_model.n_evidence == 300
    acq_x = bolfi.target_model._gp.X
    # check_inference_with_informative_data(res, 1, true_params, error_bound=.2)
    assert np.abs(res['samples']['t1'] - true_params['t1']) < 0.2
    assert np.abs(res['samples']['t2'] - true_params['t2']) < 0.2

    # Test that you can continue the inference where we left off
    res = bolfi.infer(310)
    assert bolfi.target_model.n_evidence == 310
    assert np.array_equal(bolfi.target_model._gp.X[:300,:], acq_x)

    post = bolfi.infer_posterior()

    post_ml, _ = post.ML
    post_map, _ = post.MAP
    vals_ml = dict(t1=np.array([post_ml[0]]), t2=np.array([post_ml[1]]))
    check_inference_with_informative_data(vals_ml, 1, true_params, error_bound=.2)
    vals_map = dict(t1=np.array([post_map[0]]), t2=np.array([post_map[1]]))
    check_inference_with_informative_data(vals_map, 1, true_params, error_bound=.2)

    # TODO: this is very, very slow in Travis???
    # n_samples = 100
    # n_chains = 4
    # res_sampling = bolfi.sample(n_samples, n_chains=n_chains)
    # check_inference_with_informative_data(res_sampling.samples, n_samples//2*n_chains, true_params, error_bound=.2)

    # check the cached predictions for RBF
    x = np.random.random((1, len(true_params)))
    bolfi.target_model.is_sampling = True

    pred_mu, pred_var = bolfi.target_model._gp.predict(x)
    pred_cached_mu, pred_cached_var = bolfi.target_model.predict(x)
    assert(np.allclose(pred_mu, pred_cached_mu))
    assert(np.allclose(pred_var, pred_cached_var))

    grad_mu, grad_var = bolfi.target_model._gp.predictive_gradients(x)
    grad_cached_mu, grad_cached_var = bolfi.target_model.predictive_gradients(x)
    assert(np.allclose(grad_mu, grad_cached_mu))
    assert(np.allclose(grad_var, grad_cached_var))
