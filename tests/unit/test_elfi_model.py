import pytest

import numpy as np
import scipy.stats as ss

import elfi.model.elfi_model as em


def test_node_reference_str():
    # This is important because it is used when passing NodeReferences as InferenceMethod
    # arguments
    ref = em.NodeReference('test')
    assert str(ref) == 'test'