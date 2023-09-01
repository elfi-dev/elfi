import pytest

import numpy as np

import elfi
from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.classifier import LogisticRegression
from elfi.model.extensions import ModelPrior


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


def test_bolfire_init(true_param, seed):
    # define the simple gaussian elfi model
    m = simple_gaussian_model(true_param, seed)

    # define the bolfire method
    bolfire_method = elfi.BOLFIRE(model=m, n_training_data=10)

    # check the size of mariginal data (should be the size of training data x number of summaries)
    assert bolfire_method.marginal.shape == (10, 10)
    # check the feature names
    assert bolfire_method.feature_names == [f'power_{i}' for i in range(10)]
    # check the type of a default classifier
    assert isinstance(bolfire_method.classifier, LogisticRegression)
    # check the length of observed feature values
    assert len(bolfire_method.observed[0]) == 10
    # check the type of the prior
    assert isinstance(bolfire_method.prior, ModelPrior)
    # check the acquisition function and GP regression related parameters
    assert bolfire_method.bounds is None
    assert bolfire_method.acq_noise_var == 0
    assert bolfire_method.exploration_rate == 10
    assert bolfire_method.update_interval == 1
    assert bolfire_method.n_initial_evidence == 0
    assert isinstance(bolfire_method.acquisition_method, LCBSC)


@pytest.mark.slowtest
def test_bolfire(true_param, seed):
    # define the simple gaussian elfi model
    m = simple_gaussian_model(true_param, seed)

    # define the bolfire method
    bolfire_method = elfi.BOLFIRE(
        model=m,
        n_training_data=500,
        n_initial_evidence=10,
        update_interval=1,
        bounds={'mu': (-5, 5)},
    )

    # run inference
    n_evidence = 100
    bolfire_posterior = bolfire_method.fit(n_evidence, bar=False)

    # check the number of evidence
    assert bolfire_method.n_evidence == n_evidence

    # check the map estimates
    map_estimates = bolfire_posterior.compute_map_estimates()
    assert np.abs(map_estimates['mu'] - true_param) <= 0.5

    # run sampling
    n_samples = 400
    bolfire_sample = bolfire_method.sample(n_samples)

    # check the sample (posterior) means
    sample_means = bolfire_sample.sample_means
    assert np.abs(sample_means['mu'] - true_param) <= 1.5
