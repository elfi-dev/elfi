import pytest

import numpy as np

import elfi


def simple_gaussian_model(true_param, seed, n_summaries=10):
    """The simple gaussian model that has been used as a toy example in the LFIRE paper."""

    def power(x, y):
        return x**y

    m = elfi.ElfiModel()
    mu = elfi.Prior('uniform', -5, 10, model=m, name='mu')
    y = elfi.Simulator(gauss, *[mu], observed=gauss(true_param, seed=seed), name='y')
    for i in range(n_summaries):
        elfi.Summary(power, y, i, model=m, name=f'power_{i}')
    return m


def gauss(mu, sigma=3, n_obs=1, batch_size=1, seed=None, *args, **kwargs):
    if isinstance(seed, int):
        np.random.seed(seed)
    mu = np.asanyarray(mu).reshape((-1, 1))
    sigma = np.asanyarray(sigma).reshape((-1, 1))
    return np.random.normal(mu, sigma, size=(batch_size, n_obs))


@pytest.fixture
def true_param():
    return 2.6


@pytest.fixture
def seed():
    return 4


@pytest.fixture
def parameter_values():
    return {'mu': 1.0}


@pytest.fixture
def bolfire_method(true_param, seed):
    m = simple_gaussian_model(true_param, seed)
    return elfi.BOLFIRE(m, 10)


def test_generate_marginal(bolfire_method):
    assert bolfire_method._generate_marginal().shape == (10, 10)


def test_generate_training_data(bolfire_method, parameter_values):
    likelihood = np.random.rand(10, 10)
    X, y = bolfire_method._generate_training_data(likelihood, bolfire_method.marginal)
    assert X.shape == (20, 10)
    assert y.shape == (20,)
