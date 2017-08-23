import numpy as np
import pytest
import scipy.stats as ss

import elfi.distributions as dst


def line(a, x, b):
    return a*x + b


def test_endpoint_calculation():
    a, b = np.random.rand(2)
    x = np.random.rand(2)
    y = line(a, x, b)

    assert np.allclose(dst._low(x, y), -b/a)
    assert np.allclose(dst._high(x, y), (1-b)/a)


def test_that_interpolation_is_piecewise():
    X = np.random.rand(100)
    cdf = dst.ecdf(X)

    # test scalar args
    assert cdf(-100) == 0
    assert cdf(0.5) > 0
    assert cdf(100) == 1

    assert all(cdf(np.linspace(-10, -1, 3)) == np.array([0, 0, 0]))
    assert all(cdf(np.linspace(0, 1, 3)) >= 0)
    assert all(cdf(np.linspace(1, 10, 3)) == np.array([1, 1, 1]))


def test_eppf_domain():
    X = np.random.rand(100)
    ppf = dst.eppf(X)
    with pytest.raises(ValueError):
        ppf(-1)

    with pytest.raises(ValueError):
        ppf(2)


def cov2corr(cov):
    """Convert a covariance matrix into a correlation matrix.

    Parameters
    ----------
    cov : np.ndarray
        a covariance matrix
    """
    std = np.sqrt(np.diag(cov))[:, np.newaxis]
    return cov / std.T / std


# A multivariate Gaussian can be written as a meta-Gaussian
def test_metagaussian_iid_normal():
    p = np.random.randint(2, 10)
    cov = np.eye(p)

    mvn = ss.multivariate_normal(cov=cov)
    marginals = [ss.norm(0, 1) for i in range(p)]
    mg = dst.MetaGaussian(corr=cov2corr(cov), marginals=marginals)

    theta = mvn.rvs()
    assert np.allclose(mvn.logpdf(theta), mg.logpdf(theta))
    assert np.allclose(mvn.pdf(theta), mg.pdf(theta))

    Theta = mvn.rvs(3)
    assert np.allclose(mvn.logpdf(Theta), mg.logpdf(Theta))
    assert np.allclose(mvn.pdf(Theta), mg.pdf(Theta))


def test_metagaussian_with_covariance():
    p = np.random.randint(2, 10)
    a = np.random.rand()
    df = p + 10*np.random.rand()
    cov = ss.invwishart.rvs(scale=a*np.eye(p), df=df)
    stds = np.sqrt(np.diag(cov))

    mean = 10*np.random.rand(p)
    mvn = ss.multivariate_normal(mean=mean, cov=cov)

    marginals = [ss.norm(mean[i], stds[i]) for i in range(p)]
    mg = dst.MetaGaussian(corr=cov2corr(cov), marginals=marginals)

    theta = mvn.rvs()
    assert np.allclose(mvn.logpdf(theta), mg.logpdf(theta))
    assert np.allclose(mvn.pdf(theta), mg.pdf(theta))

    Theta = mvn.rvs(3)
    assert np.allclose(mvn.logpdf(Theta), mg.logpdf(Theta))
    assert np.allclose(mvn.pdf(Theta), mg.pdf(Theta))
