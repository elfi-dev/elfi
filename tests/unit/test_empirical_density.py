import numpy as np
import pytest

import elfi.methods.empirical_density as ed


def line(a, x, b):
    return a*x + b


def test_endpoint_calculation():
    a, b = np.random.rand(2)
    x = np.random.rand(2)
    y = line(a, x, b)

    assert np.allclose(ed._low(x, y), -b/a)
    assert np.allclose(ed._high(x, y), (1-b)/a)


def test_that_interpolation_is_piecewise():
    X = np.random.rand(100)
    cdf = ed.ecdf(X)

    # test scalar args
    assert cdf(-100) == 0
    assert cdf(0.5) > 0
    assert cdf(100) == 1

    assert all(cdf(np.linspace(-10, -1, 3)) == np.array([0, 0, 0]))
    assert all(cdf(np.linspace(0, 1, 3)) >= 0)
    assert all(cdf(np.linspace(1, 10, 3)) == np.array([1, 1, 1]))


def test_ppf_domain():
    X = np.random.rand(100)
    ppf = ed.ppf(X)
    with pytest.raises(ValueError):
        ppf(-1)

    with pytest.raises(ValueError):
        ppf(2)
