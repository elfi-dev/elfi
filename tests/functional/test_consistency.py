import logging

import numpy as np
import pytest

import elfi

"""This module tests the consistency of results when using the same seed."""


def check_consistent_sample(sample, sample_diff, sample_same):
    assert not np.array_equal(sample.outputs['t1'], sample_diff.outputs['t1'])

    assert np.allclose(sample.outputs['t1'], sample_same.outputs['t1'])
    assert np.allclose(sample.outputs['t2'], sample_same.outputs['t2'])

    # BOLFI does not have d in its outputs
    if 'd' in sample.outputs:
        assert np.allclose(sample.outputs['d'], sample_same.outputs['d'])


@pytest.mark.usefixtures('with_all_clients')
def test_rejection(ma2):
    bs = 3
    n_samples = 3
    n_sim = 9

    rej = elfi.Rejection(ma2, 'd', batch_size=bs)
    sample = rej.sample(n_samples, n_sim=n_sim)
    seed = rej.seed

    rej = elfi.Rejection(ma2, 'd', batch_size=bs)
    sample_diff = rej.sample(n_samples, n_sim=n_sim)

    rej = elfi.Rejection(ma2, 'd', batch_size=bs, seed=seed)
    sample_same = rej.sample(n_samples, n_sim=n_sim)

    check_consistent_sample(sample, sample_diff, sample_same)


@pytest.mark.usefixtures('with_all_clients')
def test_smc(ma2):
    bs = 3
    n_samples = 10
    thresholds = [1, .9, .8]

    smc = elfi.SMC(ma2, 'd', batch_size=bs)
    sample = smc.sample(n_samples, thresholds=thresholds)
    seed = smc.seed

    smc = elfi.SMC(ma2, 'd', batch_size=bs, seed=seed)
    sample_same = smc.sample(n_samples, thresholds=thresholds)

    smc = elfi.SMC(ma2, 'd', batch_size=bs)
    sample_diff = smc.sample(n_samples, thresholds=thresholds)

    check_consistent_sample(sample, sample_diff, sample_same)


@pytest.mark.usefixtures('with_all_clients')
def test_adaptive_distance_smc(ma2):
    bs = 3
    n_samples = 10
    rounds = 3
    quantile = 0.9

    # use adaptive distance:
    ma2['d'].become(elfi.AdaptiveDistance(ma2['S1'], ma2['S2']))

    ad_smc = elfi.AdaptiveDistanceSMC(ma2, 'd', batch_size=bs)
    sample = ad_smc.sample(n_samples, rounds, quantile=quantile)
    seed = ad_smc.seed

    ad_smc = elfi.AdaptiveDistanceSMC(ma2, 'd', batch_size=bs, seed=seed)
    sample_same = ad_smc.sample(n_samples, rounds, quantile=quantile)

    ad_smc = elfi.AdaptiveDistanceSMC(ma2, 'd', batch_size=bs)
    sample_diff = ad_smc.sample(n_samples, rounds, quantile=quantile)

    check_consistent_sample(sample, sample_diff, sample_same)


@pytest.mark.usefixtures('with_all_clients')
def test_bo(ma2):
    bs = 2
    upd_int = 1
    n_evi = 16
    init_evi = 10
    bounds = {'t1': (-2, 2), 't2': (-1, 1)}
    anv = .1

    bo = elfi.BayesianOptimization(
        ma2,
        'd',
        initial_evidence=init_evi,
        update_interval=upd_int,
        batch_size=bs,
        bounds=bounds,
        acq_noise_var=anv)
    res = bo.infer(n_evidence=n_evi)
    seed = bo.seed

    bo = elfi.BayesianOptimization(
        ma2,
        'd',
        seed=seed,
        initial_evidence=init_evi,
        update_interval=upd_int,
        batch_size=bs,
        bounds=bounds,
        acq_noise_var=anv)
    res_same = bo.infer(n_evidence=n_evi)

    bo = elfi.BayesianOptimization(
        ma2,
        'd',
        initial_evidence=init_evi,
        update_interval=upd_int,
        batch_size=bs,
        bounds=bounds,
        acq_noise_var=anv)
    res_diff = bo.infer(n_evidence=n_evi)

    check_consistent_sample(res, res_diff, res_same)

    assert not np.array_equal(res.x_min, res_diff.x_min)
    assert np.allclose(res.x_min['t1'], res_same.x_min['t1'], atol=1e-07)
    assert np.allclose(res.x_min['t2'], res_same.x_min['t2'], atol=1e-07)


# TODO: skipped in travis due to NUTS initialization failing too often. Should be fixed.
@pytest.mark.usefixtures('with_all_clients', 'skip_travis')
def test_bolfi(ma2):
    bs = 2
    n_samples = 4
    upd_int = 1
    n_evi = 16
    init_evi = 10
    bounds = {'t1': (-2, 2), 't2': (-1, 1)}
    anv = .1
    nchains = 2

    bolfi = elfi.BOLFI(
        ma2,
        'd',
        initial_evidence=init_evi,
        update_interval=upd_int,
        batch_size=bs,
        bounds=bounds,
        acq_noise_var=anv)
    sample = bolfi.sample(n_samples, n_evidence=n_evi, n_chains=nchains)
    seed = bolfi.seed

    bolfi = elfi.BOLFI(
        ma2,
        'd',
        initial_evidence=init_evi,
        update_interval=upd_int,
        batch_size=bs,
        bounds=bounds,
        acq_noise_var=anv)
    sample_diff = bolfi.sample(n_samples, n_evidence=n_evi, n_chains=nchains)

    bolfi = elfi.BOLFI(
        ma2,
        'd',
        seed=seed,
        initial_evidence=init_evi,
        update_interval=upd_int,
        batch_size=bs,
        bounds=bounds,
        acq_noise_var=anv)
    sample_same = bolfi.sample(n_samples, n_evidence=n_evi, n_chains=nchains)

    check_consistent_sample(sample, sample_diff, sample_same)


def test_bsl(ma2):
    bs = 500
    n_samples = 3

    bsl_res = elfi.BSL(ma2, bs, batch_size=bs)
    sample = bsl_res.sample(n_samples, sigma_proposals=np.eye(2))
    seed = bsl_res.seed

    bsl_res = elfi.BSL(ma2, bs, batch_size=bs)
    sample_diff = bsl_res.sample(n_samples, sigma_proposals=np.eye(2))

    bsl_res = elfi.BSL(ma2, bs, batch_size=bs, seed=seed)
    sample_same = bsl_res.sample(n_samples, sigma_proposals=np.eye(2))

    check_consistent_sample(sample, sample_diff, sample_same)

