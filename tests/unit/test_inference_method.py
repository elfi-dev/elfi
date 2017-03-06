import pytest

import numpy as np
import scipy.stats as ss

import elfi
from elfi.methods.methods import InferenceMethod
from elfi.native_client import Client
from elfi.loader import get_sub_seed


def test_no_model_parameters(simple_model):
    simple_model.parameters = None

    with pytest.raises(Exception):
        InferenceMethod(simple_model)