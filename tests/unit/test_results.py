import pytest

from elfi.methods.results import *


def test_sample():
    n_samples = 10
    parameter_names = ['a', 'b']
    distance_name = 'dist'
    samples = [
        np.random.random(n_samples),
        np.random.random(n_samples),
        np.random.random(n_samples)
    ]
    outputs = dict(zip(parameter_names + [distance_name], samples))
    sample = Sample(
        method_name="TestRes",
        outputs=outputs,
        parameter_names=parameter_names,
        discrepancy_name=distance_name,
        something='x',
        something_else='y',
        n_sim=0, )

    assert sample.method_name == "TestRes"
    assert hasattr(sample, 'samples')
    assert sample.n_samples == n_samples
    assert sample.dim == len(parameter_names)

    assert np.allclose(samples[0], sample.samples_array[:, 0])
    assert np.allclose(samples[1], sample.samples_array[:, 1])
    assert np.allclose(samples[-1], sample.discrepancies)

    assert hasattr(sample, 'something')
    assert sample.something_else == 'y'

    with pytest.raises(AttributeError):
        sample.not_here

    # Test summary
    sample.summary()


def test_bolfi_sample():
    n_chains = 3
    n_iters = 10
    warmup = 5
    parameter_names = ['a', 'b']
    chains = np.random.random((n_chains, n_iters, len(parameter_names)))

    result = BolfiSample(
        method_name="TestRes",
        chains=chains,
        parameter_names=parameter_names,
        warmup=warmup,
        something='x',
        something_else='y',
        n_sim=0, )

    assert result.method_name == "TestRes"
    assert hasattr(result, 'samples')
    assert hasattr(result, 'chains')
    assert hasattr(result, 'outputs')
    assert result.n_samples == n_chains * (n_iters - warmup)
    assert result.dim == len(parameter_names)

    # verify that chains are merged correctly
    s0 = np.concatenate([chains[i, warmup:, 0] for i in range(n_chains)])
    s1 = np.concatenate([chains[i, warmup:, 1] for i in range(n_chains)])
    assert np.allclose(s0, result.samples[parameter_names[0]])
    assert np.allclose(s1, result.samples[parameter_names[1]])

    assert hasattr(result, 'something')
    assert result.something_else == 'y'
