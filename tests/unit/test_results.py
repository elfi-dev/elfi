import pytest

from elfi.methods.results import *
from elfi.methods.posteriors import BolfiPosterior


def test_Result():
    n_samples = 10
    parameter_names = ['a', 'b']
    distance_name = 'dist'
    samples = [np.random.random(n_samples), np.random.random(n_samples), np.random.random(n_samples)]
    outputs = dict(zip(parameter_names + [distance_name], samples))
    result = Result(method_name="TestRes",
                    outputs=outputs,
                    parameter_names=parameter_names,
                    discrepancy_name=distance_name,
                    something='x',
                    something_else='y'
                    )

    assert result.method_name == "TestRes"
    assert hasattr(result, 'samples')
    assert result.n_samples == n_samples
    assert result.n_params == len(parameter_names)

    assert np.allclose(samples[:-1], result.samples_list)
    assert np.allclose(samples[-1], result.discrepancy)

    assert hasattr(result, 'something')
    assert result.something_else == 'y'

    with pytest.raises(AttributeError):
        result.not_here


def test_ResultBOLFI():
    n_chains = 3
    n_iters = 10
    warmup = 5
    parameter_names = ['a', 'b']
    chains = np.random.random((n_chains, n_iters, len(parameter_names)))

    result = ResultBOLFI(method_name="TestRes",
                         chains=chains,
                         parameter_names=parameter_names,
                         warmup=warmup,
                         something='x',
                         something_else='y'
                         )

    assert result.method_name == "TestRes"
    assert hasattr(result, 'samples')
    assert hasattr(result, 'chains')
    assert hasattr(result, 'outputs')
    assert result.n_samples == n_chains * (n_iters - warmup)
    assert result.n_params == len(parameter_names)

    # verify that chains are merged correctly
    s0 = np.concatenate([chains[i, warmup:, 0] for i in range(n_chains)])
    s1 = np.concatenate([chains[i, warmup:, 1] for i in range(n_chains)])
    assert np.allclose(s0, result.samples[parameter_names[0]])
    assert np.allclose(s1, result.samples[parameter_names[1]])

    assert hasattr(result, 'something')
    assert result.something_else == 'y'


def test_bolfi_posterior(ma2):
    m = ma2.get_model()
    #prior =