import numpy as np
import scipy.stats as ss

import elfi.methods.copula as cop
from elfi.methods.utils import cov2corr


# A multivariate Gaussian can be written as a meta-Gaussian
def test_metagaussian_iid_normal():
    p = np.random.randint(2, 10)
    cov = np.eye(p)

    mvn = ss.multivariate_normal(cov=cov)
    marginals = [ss.norm(0, 1) for i in range(p)]
    mg = cop.MetaGaussian(cov=cov, marginals=marginals)
    mg2 = cop.MetaGaussian(corr=cov2corr(cov), marginals=marginals)

    theta = mvn.rvs()
    assert np.allclose(mvn.logpdf(theta), mg.logpdf(theta))
    assert np.allclose(mvn.pdf(theta), mg.pdf(theta))

    assert np.allclose(mvn.logpdf(theta), mg2.logpdf(theta))
    assert np.allclose(mvn.pdf(theta), mg2.pdf(theta))

    Theta = mvn.rvs(3)
    assert np.allclose(mvn.logpdf(Theta), mg.logpdf(Theta))
    assert np.allclose(mvn.pdf(Theta), mg.pdf(Theta))

    assert np.allclose(mvn.logpdf(Theta), mg2.logpdf(Theta))
    assert np.allclose(mvn.pdf(Theta), mg2.pdf(Theta))


def test_metagaussian_with_covariance():
    p = np.random.randint(2, 10)
    a = np.random.rand()
    df = (p-1) + 10*np.random.rand()
    cov = ss.invwishart.rvs(scale=a*np.eye(p), df=df)
    stds = np.sqrt(np.diag(cov))

    mean = 10*np.random.rand(p)
    mvn = ss.multivariate_normal(mean=mean, cov=cov)

    marginals = [ss.norm(mean[i], stds[i]) for i in range(p)]
    mg = cop.MetaGaussian(cov=cov, marginals=marginals)
    mg2 = cop.MetaGaussian(corr=cov2corr(cov), marginals=marginals)

    theta = mvn.rvs()
    assert np.allclose(mvn.logpdf(theta), mg.logpdf(theta))
    assert np.allclose(mvn.pdf(theta), mg.pdf(theta))

    assert np.allclose(mvn.logpdf(theta), mg2.logpdf(theta))
    assert np.allclose(mvn.pdf(theta), mg2.pdf(theta))

    Theta = mvn.rvs(3)
    assert np.allclose(mvn.logpdf(Theta), mg.logpdf(Theta))
    assert np.allclose(mvn.pdf(Theta), mg.pdf(Theta))

    assert np.allclose(mvn.logpdf(Theta), mg2.logpdf(Theta))
    assert np.allclose(mvn.pdf(Theta), mg2.pdf(Theta))


def test_metagaussian_sampling_with_cov():
    rho = 0.5
    cov = np.array([[1, rho],
                     [rho, 1]])
    marginals = [ss.beta(5, 2),
                 ss.gamma(2, 2)]

    mg = cop.MetaGaussian(cov=cov, marginals=marginals)
    X = mg.rvs(3)
    assert np.all(X > 0)


def test_sliced_summary():
    idx_spec = [{2, 3}, [2, 3]]
    data = np.array([np.arange(10),
                     np.arange(10)])
    data2 = np.array([np.arange(10)])

    for idx in idx_spec:
        ss = cop.sliced_summary(idx)
        assert np.all(ss(data) == np.array([[2, 3],
                                            [2, 3]]))
        assert np.all(ss(data2) == np.array([[2, 3]]))

    ss2 = cop.sliced_summary(1)
    assert np.all(ss2(data) == np.array([[1],
                                        [1]]))
    assert np.all(ss2(data2) == np.array([[1]]))


def test_complete_informative_indices():
    informative_indices = {i: i for i in range(3)}
    res = cop.complete_informative_indices(informative_indices)
    expected = {0: 0, 1: 1, 2: 2,
                (0, 1): {0, 1},
                (0, 2): {0, 2},
                (1, 2): {1, 2}}
    assert res == expected


def test_complete_informative_indices2():
    informative_indices = {i: i for i in range(3)}
    informative_indices[1] = {0, 1}
    res = cop.complete_informative_indices(informative_indices)
    expected = {0: 0, 1: {0, 1}, 2: 2,
                (0, 1): {0, 1},
                (0, 2): {0, 2},
                (1, 2): {0, 1, 2}}
    assert res == expected
