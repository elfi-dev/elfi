import pytest
import logging

import numpy as np

from elfi.results.result import Result


def test_result_obj():
    n_samples = 10
    parameter_names = ['a', 'b']
    distance_name = 'dist'
    samples = [np.empty(n_samples), np.empty(n_samples), np.empty(n_samples)]
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
