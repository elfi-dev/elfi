from functools import partial

import numpy as np
import pytest

import elfi
from elfi.examples.gauss import euclidean_multidim, gauss_nd_mean


def toy_sim(*mu, cov, batch_size=1, random_state=None):
    out = gauss_nd_mean(*mu, cov_matrix=cov, batch_size=batch_size, random_state=random_state)
    failed = np.sqrt(np.sum(np.power(mu, 2), axis=0)) < 2.5
    out[failed] = np.nan
    return out


def get_model(true_params, seed):
    cov = (0.1 * np.diag(np.ones((2,))) + 0.5 * np.ones((2, 2)))
    sim = partial(toy_sim, cov=cov)

    random_state = np.random.RandomState(seed)
    obs = sim(*true_params, random_state=random_state)

    m = elfi.ElfiModel()
    mu_1 = elfi.Prior('uniform', 0, 5, model=m)
    mu_2 = elfi.Prior('uniform', 0, 5, model=m)
    y = elfi.Simulator(sim, mu_1, mu_2, observed=obs)
    mean = elfi.Summary(partial(np.mean, axis=1), y)
    d = elfi.Discrepancy(euclidean_multidim, mean)
    return m


@pytest.mark.slowtest
def test_failure_robust_BOLFI():
    seed = 123
    true_params = [3, 3]
    m = get_model(true_params, seed)
    bounds = {'mu_1': (0, 5), 'mu_2': (0, 5)}
    target_model = elfi.RobustGPyRegression(m.parameter_names, bounds=bounds, thd=0.5)
    bolfi = elfi.BOLFI(m['d'], initial_evidence=20, target_model=target_model, seed=seed)
    post = bolfi.fit(n_evidence=100)

    # check that optimisation avoided infeasible parameter combinations
    assert bolfi.target_model.n_valid_evidence > 90

    # check model minimum
    res_1 = bolfi.extract_result()
    assert np.isclose(res_1.x_min['mu_1'], 3, atol=0.35)
    assert np.isclose(res_1.x_min['mu_2'], 3, atol=0.35)

    # check posterior mean
    res_2 = bolfi.sample(500)
    assert np.isclose(np.mean(res_2.samples['mu_1']), 3, atol=0.35)
    assert np.isclose(np.mean(res_2.samples['mu_2']), 3, atol=0.35)
