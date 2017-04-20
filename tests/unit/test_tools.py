import pytest
import numpy as np

import elfi


def test_vectorize_decorator():
    batch_size = 3
    a = np.array([1,2,3])
    b = np.array([3,2,1])

    @elfi.tools.vectorize
    def simulator(a, b, random_state=None):
        return a*b

    assert np.array_equal(a * b, simulator(a, b, batch_size=batch_size))

    @elfi.tools.vectorize(constants=1)
    def simulator(a, constant, random_state=None):
        return a*constant

    assert np.array_equal(a * 5, simulator(a, 5, batch_size=batch_size))

    @elfi.tools.vectorize(1)
    def simulator(a, constant, random_state=None):
        return a*constant

    assert np.array_equal(a * 5, simulator(a, 5, batch_size=batch_size))

    @elfi.tools.vectorize(constants=(0,2))
    def simulator(constant0, b, constant2, random_state=None):
        return constant0*b*constant2

    assert np.array_equal(2 * b * 7, simulator(2, b, 7, batch_size=batch_size))

    with pytest.raises(ValueError):
        simulator(2, b, 7, batch_size=2*batch_size)
