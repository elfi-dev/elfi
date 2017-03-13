import pytest

from elfi.methods.methods import InferenceMethod


def test_no_model_parameters(simple_model):
    simple_model.parameters = None

    with pytest.raises(Exception):
        InferenceMethod(simple_model)