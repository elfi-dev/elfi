from functools import partial

import numpy as np
import pytest

import elfi
import elfi.methods.post_processing as pp
from elfi.examples import gauss, ma2
from elfi.methods.post_processing import LinearAdjustment, adjust_posterior


def _statistics(arr):
    return arr.mean(), arr.var()


def test_get_adjustment():
    with pytest.raises(ValueError):
        pp._get_adjustment('doesnotexist')


def test_single_parameter_linear_adjustment():
    """A regression test against values obtained in the notebook."""
    seed = 20170616
    n_obs = 50
    batch_size = 1000
    mu, sigma = (5, 1)

    # Hyperparameters
    mu0, sigma0 = (10, 100)

    y_obs = gauss.gauss(
        mu, sigma, n_obs=n_obs, batch_size=1, random_state=np.random.RandomState(seed))
    sim_fn = partial(gauss.gauss, sigma=sigma, n_obs=n_obs)

    # Posterior
    n = y_obs.shape[1]
    mu1 = (mu0 / sigma0**2 + y_obs.sum() / sigma**2) / (1 / sigma0**2 + n / sigma**2)
    sigma1 = (1 / sigma0**2 + n / sigma**2)**(-0.5)

    # Model
    m = elfi.ElfiModel()
    elfi.Prior('norm', mu0, sigma0, model=m, name='mu')
    elfi.Simulator(sim_fn, m['mu'], observed=y_obs, name='gauss')
    elfi.Summary(lambda x: x.mean(axis=1), m['gauss'], name='ss_mean')
    elfi.Distance('euclidean', m['ss_mean'], name='d')

    res = elfi.Rejection(m['d'], output_names=['ss_mean'], batch_size=batch_size,
                         seed=seed).sample(1000, threshold=1)
    adj = elfi.adjust_posterior(model=m, sample=res, parameter_names=['mu'], summary_names=['ss_mean'])

    assert np.allclose(_statistics(adj.outputs['mu']), (4.9772879640569778, 0.02058680115402544))


# TODO: Use a fixture for the model
def test_nonfinite_values():
    """A regression test against values obtained in the notebook."""
    seed = 20170616
    n_obs = 50
    batch_size = 1000
    mu, sigma = (5, 1)

    # Hyperparameters
    mu0, sigma0 = (10, 100)

    y_obs = gauss.gauss(
        mu, sigma, n_obs=n_obs, batch_size=1, random_state=np.random.RandomState(seed))
    sim_fn = partial(gauss.gauss, sigma=sigma, n_obs=n_obs)

    # Posterior
    n = y_obs.shape[1]
    mu1 = (mu0 / sigma0**2 + y_obs.sum() / sigma**2) / (1 / sigma0**2 + n / sigma**2)
    sigma1 = (1 / sigma0**2 + n / sigma**2)**(-0.5)

    # Model
    m = elfi.ElfiModel()
    elfi.Prior('norm', mu0, sigma0, model=m, name='mu')
    elfi.Simulator(sim_fn, m['mu'], observed=y_obs, name='gauss')
    elfi.Summary(lambda x: x.mean(axis=1), m['gauss'], name='ss_mean')
    elfi.Distance('euclidean', m['ss_mean'], name='d')

    res = elfi.Rejection(m['d'], output_names=['ss_mean'], batch_size=batch_size,
                         seed=seed).sample(1000, threshold=1)

    # Add some invalid values
    res.outputs['mu'] = np.append(res.outputs['mu'], np.array([np.inf]))
    res.outputs['ss_mean'] = np.append(res.outputs['ss_mean'], np.array([np.inf]))

    with pytest.warns(UserWarning):
        adj = elfi.adjust_posterior(
            model=m, sample=res, parameter_names=['mu'], summary_names=['ss_mean'])

    assert np.allclose(_statistics(adj.outputs['mu']), (4.9772879640569778, 0.02058680115402544))


def test_multi_parameter_linear_adjustment():
    """A regression test against values obtained in the notebook."""
    seed = 20170511
    threshold = 0.2
    batch_size = 1000
    n_samples = 500
    m = ma2.get_model(true_params=[0.6, 0.2], seed_obs=seed)

    summary_names = ['S1', 'S2']
    parameter_names = ['t1', 't2']
    linear_adjustment = LinearAdjustment()

    res = elfi.Rejection(
        m['d'],
        batch_size=batch_size,
        output_names=['S1', 'S2'],
        # output_names=summary_names, # fails ?!?!?
        seed=seed).sample(
            n_samples, threshold=threshold)
    adjusted = adjust_posterior(
        model=m,
        sample=res,
        parameter_names=parameter_names,
        summary_names=summary_names,
        adjustment=linear_adjustment)
    t1 = adjusted.outputs['t1']
    t2 = adjusted.outputs['t2']

    t1_mean, t1_var = (0.51606048286584782, 0.017253007645871756)
    t2_mean, t2_var = (0.15805189695581101, 0.028004406914362647)
    assert np.allclose(_statistics(t1), (t1_mean, t1_var))
    assert np.allclose(_statistics(t2), (t2_mean, t2_var))
