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


@pytest.fixture(scope='module')
def model_1():
    seed = 123
    true_params = [3, 3]
    m = get_model(true_params, seed)
    data = m.generate(20, seed=seed)
    X = np.column_stack((data['mu_1'], data['mu_2']))
    Y = data['d']
    bounds = {'mu_1': (0, 5), 'mu_2': (0, 5)}
    target_model = elfi.RobustGPyRegression(m.parameter_names, bounds=bounds)
    target_model.update(X, Y)
    return target_model


@pytest.fixture(scope='module')
def model_2():
    seed = 123
    true_params = [3, 3]
    m = get_model(true_params, seed)
    data = m.generate(20, seed=seed)
    X = np.column_stack((data['mu_1'], data['mu_2']))
    Y = data['d']
    bounds = {'mu_1': (0, 5), 'mu_2': (0, 5)}
    target_model = elfi.RobustGPyRegression(m.parameter_names, bounds=bounds, thd=0.5)
    target_model.update(X, Y)
    return target_model


def test_update():
    X = np.random.rand(1, 2)
    Y = np.ones((1, 1))
    target_model = elfi.RobustGPyRegression(['param_1', 'param_2'], thd=0.5)
    target_model.update(X, Y)
    assert np.all(target_model.X == X)
    assert np.all(target_model.Y == Y)


def test_cannot_update():
    X = np.random.rand(1, 2)
    Y = np.nan * np.ones((1, 1))
    target_model = elfi.RobustGPyRegression(['param_1', 'param_2'], thd=0.5)
    with pytest.raises(RuntimeError):
        target_model.update(X, Y)


@pytest.mark.parametrize('model', ['model_1', 'model_2'])
def test_n_evidence(model, request):
    target_model = request.getfixturevalue(model)
    assert target_model.n_evidence == 20
    assert target_model.n_valid_evidence < target_model.n_evidence


@pytest.mark.parametrize('model', ['model_1', 'model_2'])
def test_predict(model, request):
    target_model = request.getfixturevalue(model)
    mu1, _ = target_model.predict([3, 3])
    mu2, _ = target_model.predict([4, 4])
    assert mu1 < mu2


def test_predict_infeasible(model_2):
    target_model = model_2
    pred = target_model.predict([0, 0])
    assert pred[0] == target_model.FAILED_OUTPUT
    assert pred[1] == target_model.FAILED_VAR


def test_predict_gradients_infeasible(model_2):
    target_model = model_2
    grad = target_model.predictive_gradients([0, 0])
    assert grad[0].shape == (1, 2)
    assert grad[1].shape == (1, 2)
    assert np.all(grad[0] == 0)
    assert np.all(grad[1] == 0)


def test_success_proba(model_2):
    target_model = model_2
    prob1 = target_model.success_proba([4, 4])
    prob2 = target_model.success_proba([2, 2])
    assert prob1 > prob2


def test_success_proba_default(model_1):
    target_model = model_1
    prob = target_model.success_proba([3, 3])
    assert float(prob) == 1


def test_success_proba_gradients(model_2):
    target_model = model_2
    grad = target_model.success_proba_gradients([0, 0])
    assert grad.shape == (1, 2)


def test_success_proba_gradients_default(model_1):
    target_model = model_1
    grad = target_model.success_proba_gradients([0, 0])
    assert grad.shape == (1, 2)
    assert np.all(grad == 0)
