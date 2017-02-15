import pytest

import numpy as np
import scipy.stats as ss

import elfi
import examples.ma2 as ma2


def test_observed():
    true_params = [.6, .2]
    m = ma2.get_model(100, true_params=true_params)
    y = m.observed['MA2']
    S1 = m.get_reference('S1')

    S1_observed = ma2.autocov(y)
    assert np.array_equal(S1.observed, S1_observed)
