"""Test inference methods point estimates with informative data from the MA2 process."""

from collections import OrderedDict

import numpy as np
import pytest

import elfi
from elfi.examples import ma2
from elfi.methods.bo.utils import minimize, stochastic_optimization
from elfi.methods.bsl.pre_sample_methods import estimate_whitening_matrix, select_penalty
from elfi.methods.bsl.pdf_methods import standard_likelihood, unbiased_likelihood, semiparametric_likelihood, robust_likelihood
from elfi.model.elfi_model import NodeReference


def setup_ma2_with_informative_data(n_obs=100):
    true_params = OrderedDict([('t1', .6), ('t2', .2)])

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
    assert len(np.unique(res.discrepancies)) == N

    assert res.accept_rate == quantile
    assert res.n_sim == int(N / quantile)


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
    assert len(np.unique(res.discrepancies)) == N


@pytest.mark.usefixtures('with_all_clients')
def test_smc_with_thresholds():
    m, true_params = setup_ma2_with_informative_data()

    thresholds = [.5, .25, .1]
    N = 1000
    smc = elfi.SMC(m['d'], batch_size=20000)
    res = smc.sample(N, thresholds=thresholds)

    check_inference_with_informative_data(res.samples, N, true_params)

    # We should be able to carry out the inference in less than six batches
    assert res.populations[-1].n_batches < 6


@pytest.mark.usefixtures('with_all_clients')
def test_smc_with_quantiles():
    m, true_params = setup_ma2_with_informative_data()

    quantiles = [.5, .5, .5]
    N = 1000
    smc = elfi.SMC(m['d'], batch_size=20000)
    res = smc.sample(N, quantiles=quantiles)

    check_inference_with_informative_data(res.samples, N, true_params)


@pytest.mark.usefixtures('with_all_clients')
def test_adaptivethresholdsmc():
    m, true_params = setup_ma2_with_informative_data()

    N = 1000
    adathsmc = elfi.AdaptiveThresholdSMC(m['d'], batch_size=500)
    res = adathsmc.sample(N, max_iter=4)

    check_inference_with_informative_data(res.samples, N, true_params)

    # We should be able to carry out the inference in less than six batches
    # assert res.populations[-1].n_batches < 6
    assert len(res.populations) == 4


@pytest.mark.usefixtures('with_all_clients')
def test_adaptive_distance_smc():
    m, true_params = setup_ma2_with_informative_data()

    # use adaptive distance:
    m['d'].become(elfi.AdaptiveDistance(m['S1'], m['S2']))

    N = 1000
    rounds = 3
    ad_smc = elfi.AdaptiveDistanceSMC(m['d'], batch_size=20000)
    ad_res = ad_smc.sample(N, rounds)

    check_inference_with_informative_data(ad_res.samples, N, true_params)

    assert len(ad_res.populations) == rounds

    # We should be able to carry out the inference in less than six batches
    assert ad_res.populations[-1].n_batches < 6


@pytest.mark.slowtest
@pytest.mark.usefixtures('with_all_clients', 'skip_travis')
def test_BOLFI():
    m, true_params = setup_ma2_with_informative_data()

    # Log discrepancy tends to work better
    log_d = NodeReference(m['d'], state=dict(_operation=np.log), model=m, name='log_d')

    bolfi = elfi.BOLFI(
        log_d,
        initial_evidence=20,
        update_interval=10,
        batch_size=5,
        bounds={'t1': (-2, 2),
                't2': (-1, 1)},
        acq_noise_var=.1)
    n = 300
    res = bolfi.infer(300)
    assert bolfi.target_model.n_evidence == 300
    acq_x = bolfi.target_model._gp.X

    # check_inference_with_informative_data(res, 1, true_params, error_bound=.2)
    assert np.abs(res.x_min['t1'] - true_params['t1']) < 0.2
    assert np.abs(res.x_min['t2'] - true_params['t2']) < 0.2

    # Test that you can continue the inference where we left off
    res = bolfi.infer(n + 10)
    assert bolfi.target_model.n_evidence == n + 10
    assert np.array_equal(bolfi.target_model._gp.X[:n, :], acq_x)

    post = bolfi.extract_posterior()

    # TODO: make cleaner.
    post_ml = minimize(
        post._neg_unnormalized_loglikelihood,
        post.model.bounds,
        grad=post._gradient_neg_unnormalized_loglikelihood,
        prior=post.prior,
        n_start_points=post.n_inits,
        maxiter=post.max_opt_iters,
        random_state=post.random_state)[0]
    # TODO: Here we cannot use the minimize method due to sharp edges in the posterior.
    #       If a MAP method is implemented, one must be able to set the optimizer and
    #       provide its options.
    post_map = stochastic_optimization(post._neg_unnormalized_logposterior, post.model.bounds)[0]
    vals_ml = dict(t1=np.array([post_ml[0]]), t2=np.array([post_ml[1]]))
    check_inference_with_informative_data(vals_ml, 1, true_params, error_bound=.2)
    vals_map = dict(t1=np.array([post_map[0]]), t2=np.array([post_map[1]]))
    check_inference_with_informative_data(vals_map, 1, true_params, error_bound=.2)

    n_samples = 400
    n_chains = 4
    res_sampling = bolfi.sample(n_samples, n_chains=n_chains)
    check_inference_with_informative_data(
        res_sampling.samples, n_samples // 2 * n_chains, true_params, error_bound=.2)

    # check the cached predictions for RBF
    x = np.random.random((1, len(true_params)))
    bolfi.target_model.is_sampling = True

    pred_mu, pred_var = bolfi.target_model._gp.predict(x)
    pred_cached_mu, pred_cached_var = bolfi.target_model.predict(x)
    assert (np.allclose(pred_mu, pred_cached_mu))
    assert (np.allclose(pred_var, pred_cached_var))

    grad_mu, grad_var = bolfi.target_model._gp.predictive_gradients(x)
    grad_cached_mu, grad_cached_var = bolfi.target_model.predictive_gradients(x)
    assert (np.allclose(grad_mu[:, :, 0], grad_cached_mu))
    assert (np.allclose(grad_var, grad_cached_var))

    # test calculation of prior logpdfs
    true_logpdf_prior = ma2.CustomPrior1.logpdf(x[0, 0], 2)
    true_logpdf_prior += ma2.CustomPrior2.logpdf(x[0, 1], x[0, 0, ], 1)

    assert np.isclose(true_logpdf_prior, post.prior.logpdf(x[0, :]))


@pytest.mark.slowtest
def test_romc():
    """Test that ROMC provides sensible samples at the MA2 example."""
    model, true_params = setup_ma2_with_informative_data()

    # define romc inference method
    bounds = [(-2, 2), (-2, 2)]
    romc = elfi.ROMC(model, bounds=bounds, discrepancy_name="d")

    # solve problems
    n1 = 100
    seed = 21
    romc.solve_problems(n1=n1, seed=seed)

    # estimate regions
    eps_filter = .02
    romc.estimate_regions(eps_filter=eps_filter, fit_models=True, eps_cutoff=0.1)

    # sample from posterior
    n2 = 50
    romc.sample(n2=n2)

    romc_mean = romc.result.sample_means_array
    romc_cov = romc.result.samples_cov()

    # Inference with Rejection
    N = 10000
    rej = elfi.Rejection(model, discrepancy_name="d", batch_size=10000, seed=seed)
    result = rej.sample(N, threshold=.1)

    rejection_mean = result.sample_means_array
    rejection_cov = np.cov(result.samples_array.T)

    # assert summary statistics of samples match the ground truth
    assert np.allclose(romc_mean, rejection_mean, atol=.1)
    assert np.allclose(romc_cov, rejection_cov, atol=.1)


def check_bsl(likelihood, n_sim, error_bound=.15):
    n_obs = 50
    m, true_params = setup_ma2_with_informative_data(n_obs=n_obs)

    mcmc_iters = 2000
    est_posterior_cov = np.array([[0.02, 0.01],
                                  [0.01, 0.02]])

    feature_names = ['MA2']
    bsl = elfi.BSL(m, n_sim, feature_names, likelihood=likelihood, seed=123)
    bsl_res = bsl.sample(mcmc_iters, sigma_proposals=est_posterior_cov,
                         params0=np.array([0.6, 0.2]))

    check_inference_with_informative_data(bsl_res.samples, mcmc_iters,
                                          true_params, error_bound)


def check_rbsl(likelihood, n_sim, error_bound=.15):
    n_obs = 100  # as rbsl uses autocov
    m, true_params = setup_ma2_with_informative_data(n_obs=n_obs)

    mcmc_iters = 2000
    est_posterior_cov = np.array([[0.02, 0.01],
                                  [0.01, 0.02]])

    feature_names = ['S1', 'S2']
    rbsl = elfi.BSL(m, n_sim, feature_names, likelihood=likelihood, seed=123)
    rbsl_res = rbsl.sample(mcmc_iters, sigma_proposals=est_posterior_cov,
                           params0=np.array([0.6, 0.2]))

    check_inference_with_informative_data(rbsl_res.samples, mcmc_iters,
                                          true_params, error_bound)


def test_sbsl():
    """Test standard BSL provides sensible samples at the MA2 example."""
    likelihood = standard_likelihood()
    check_bsl(likelihood, 500)


def test_ubsl():
    """Test unbiased BSL provides sensible samples at the MA2 example."""
    likelihood = unbiased_likelihood()
    check_bsl(likelihood, 500)


@pytest.mark.slowtest
def test_semibsl():
    """Test semiBSL provides sensible samples at the MA2 example."""
    likelihood = semiparametric_likelihood()
    check_bsl(likelihood, 500)


@pytest.mark.slowtest
def test_rbslm():
    """Test R-BSL-M provides sensible samples at the MA2 example."""
    likelihood = robust_likelihood("mean")
    check_rbsl(likelihood, 12)


@pytest.mark.slowtest
def test_rbslv():
    """Test R-BSL-V provides sensible samples at the MA2 example."""
    likelihood = robust_likelihood("variance")
    check_rbsl(likelihood, 10)


def test_wbsl():
    """Test wBSL provides sensible samples at the MA2 example."""
    tmp_m, _ = setup_ma2_with_informative_data(n_obs=50)
    true_params = np.array([0.6, 0.2])
    feature_names = ['MA2']
    W = estimate_whitening_matrix(tmp_m, 5000, true_params, feature_names, seed=1)
    n_sim = 100
    shrinkage = "warton"
    penalty, std_value = select_penalty(model=tmp_m,
                                        n_sim=n_sim,
                                        theta=true_params,
                                        feature_names=feature_names,
                                        M=10,
                                        shrinkage=shrinkage,
                                        whitening=W,
                                        sigma=1.5,
                                        seed=1
                                        )
    likelihood = standard_likelihood(whitening=W, penalty=penalty, shrinkage=shrinkage)
    check_bsl(likelihood, n_sim)
