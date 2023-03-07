import numpy as np
import scipy.stats as ss

from elfi.methods.bsl import pdf_methods


def make_test_data(seed=270922):
    mean = np.array([0, 5])
    cov = np.array([[0.5, 0.2], [0.2, 0.5]])
    x_1 = mean
    x_2 = mean + 0.75
    p_1 = ss.multivariate_normal.logpdf(x_1, mean, cov)
    p_2 = ss.multivariate_normal.logpdf(x_2, mean, cov)
    nsim = 1000
    data = ss.multivariate_normal.rvs(mean, cov, nsim, random_state=seed)
    return data, (x_1, p_1), (x_2, p_2)


def test_gaussian_syn_likelihood():
    ssx, test_1, test_2 = make_test_data()
    assert np.isclose(pdf_methods.gaussian_syn_likelihood(ssx, test_1[0]), test_1[1], atol=0.1)
    assert np.isclose(pdf_methods.gaussian_syn_likelihood(ssx, test_2[0]), test_2[1], atol=0.1)


def test_gaussian_syn_likelihood_glasso():
    ssx, test_1, test_2 = make_test_data()

    p_0 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0])
    p_1 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='glasso', penalty=0)
    assert np.isclose(p_0, p_1)

    p_1 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='glasso', penalty=0.2)
    p_2 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='glasso', penalty=0.2)
    assert p_2 < p_1

    p_1 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='glasso', penalty=0.1)
    p_2 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='glasso', penalty=0.2)
    assert p_2 < p_1


def test_gaussian_syn_likelihood_warton():
    ssx, test_1, test_2 = make_test_data()

    p_0 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0])
    p_1 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='warton', penalty=0)
    assert np.isclose(p_0, p_1)

    p_1 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='warton', penalty=0.2)
    p_2 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='warton', penalty=0.2)
    assert p_2 < p_1

    p_1 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='warton', penalty=0.1)
    p_2 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='warton', penalty=0.2)
    assert p_2 < p_1


def test_semi_param_kernel_estimate():
    ssx, test_1, test_2 = make_test_data()
    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0])
    p_2 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0])
    assert np.isclose(p_1, test_1[1], atol=0.2)
    assert np.isclose(p_2, test_2[1], atol=0.2)


def test_semi_param_kernel_estimate_glasso():
    ssx, test_1, test_2 = make_test_data()

    p_0 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0])
    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='glasso', penalty=0)
    assert np.isclose(p_0, p_1)

    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='glasso', penalty=0.2)
    p_2 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0], shrinkage='glasso', penalty=0.2)
    assert p_2 < p_1

    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0], shrinkage='glasso', penalty=0.1)
    p_2 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0], shrinkage='glasso', penalty=0.2)
    assert p_2 < p_1


def test_semi_param_kernel_estimate_warton():
    ssx, test_1, test_2 = make_test_data()

    p_0 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0])
    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='warton', penalty=0)
    assert np.isclose(p_0, p_1)

    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='warton', penalty=0.2)
    p_2 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0], shrinkage='warton', penalty=0.2)
    assert p_2 < p_1

    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0], shrinkage='warton', penalty=0.1)
    p_2 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0], shrinkage='warton', penalty=0.2)
    assert p_2 < p_1
