import numpy as np
import scipy.stats as ss

import elfi.methods.copula as cop
from elfi.methods.utils import cov2corr


def test_metagaussian():
    p = np.random.randint(2, 10)
    cov = np.eye(p)

    mvn = ss.multivariate_normal(cov=cov)
    marginals = [ss.norm(0, 1) for i in range(p)]
    mg = cop.GaussianCopula(cov=cov, marginals=marginals)
    mg2 = cop.GaussianCopula(corr=cov2corr(cov), marginals=marginals)

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


def test_make_union():
    informative_indices = {i: i for i in range(3)}
    res = cop.make_union(informative_indices)
    expected = {0: 0, 1: 1, 2: 2,
                (0, 1): {0, 1},
                (0, 2): {0, 2},
                (1, 2): {1, 2}}
    assert res == expected


def test_make_union2():
    informative_indices = {i: i for i in range(3)}
    informative_indices[1] = {0, 1}
    res = cop.make_union(informative_indices)
    expected = {0: 0, 1: {0, 1}, 2: 2,
                (0, 1): {0, 1},
                (0, 2): {0, 2},
                (1, 2): {0, 1, 2}}
    assert res == expected


# def test_metagaussian():
#     # the gaussian distribution is a special case of a meta-Gaussian
#     p = 5
#     a = np.random.rand()
#     df = (p-1) + 10*np.random.rand()
#     cov = ss.invwishart.rvs(scale=a*np.eye(p), df=df)

#     mean = 10*np.random.rand(p)
#     mvn = ss.multivariate_normal(mean=mean, cov=cov)

#     marginals = [ss.norm(0, 1) for i in range(p)]
#     mg = cop.GaussianCopula(cov=cov, marginals=marginals)
#     mg2 = cop.GaussianCopula(corr=cov2corr(cov), marginals=marginals)

#     theta = mvn.rvs()
#     assert mvn.logpdf(theta) == mg.logpdf(theta)
#     assert mvn.pdf(theta) == mg.pdf(theta)

#     assert mvn.logpdf(theta) == mg2.logpdf(theta)
#     assert mvn.pdf(theta) == mg2.pdf(theta)
