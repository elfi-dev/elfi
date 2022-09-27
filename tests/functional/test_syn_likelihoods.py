OAimport numpy as np
import scipy.stats as ss

from elfi.methods.bsl import pdf_methods

def make_test_data(seed=270922):
    mean = [0, 5]
    cov = [[0.5, 0.2], [0.2, 0.5]]
    x_1 = [1, 5]
    x_2 = [2, 2]
    p_1 = ss.multivariate_normal.logpdf(x_1, mean, cov)
    p_2 = ss.multivariate_normal.logpdf(x_2, mean, cov)
    nsim = 500
    data = ss.multivariate_normal.rvs(mean, cov, nsim, random_state=seed)
    return data, (x_1, p_1), (x_2, p_2)

def test_gaussian_syn_likelihood():
    ssx, test_1, test_2 = make_test_data()
    assert np.isclose(pdf_methods.gaussian_syn_likelihood(ssx, test_1[0]), test_1[1], atol=0.1)
    assert np.isclose(pdf_methods.gaussian_syn_likelihood(ssx, test_2[0]), test_2[1], atol=0.5)

def test_gaussian_syn_likelihood_glasso():
    ssx, test_1, test_2 = make_test_data()

    p_10 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0])
    p_11 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='glasso', penalty=0)
    assert np.isclose(p_10, p_11)

    p_12 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='glasso', penalty=0.2)
    p_22 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='glasso', penalty=0.2)
    assert p_12 > p_11
    assert p_12 > p_22

def test_gaussian_syn_likelihood_warton():
    ssx, test_1, test_2 = make_test_data()

    p_10 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0])
    p_11 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='warton', penalty=1)
    assert np.isclose(p_10, p_11)

    p_12 = pdf_methods.gaussian_syn_likelihood(ssx, test_1[0], shrinkage='warton', penalty=0.8)
    p_22 = pdf_methods.gaussian_syn_likelihood(ssx, test_2[0], shrinkage='warton', penalty=0.8)
    assert p_12 > p_11
    assert p_12 > p_22

def test_semi_param_kernel_estimate():
    ssx, test_1, test_2 = make_test_data()
    p_1 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0])
    p_2 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0])
    assert np.isclose(p_1, test_1[1], atol=0.1)
    assert p_1 > p_2

def test_semi_param_kernel_estimate_glasso():
    ssx, test_1, test_2 = make_test_data()

    p_10 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0])
    p_11 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='glasso', penalty=0)
    assert np.isclose(p_10, p_11)

    p_12 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='glasso', penalty=0.2)
    p_22 = pdf_methods.semi_param_kernel_estimate(ssx, test_2[0], shrinkage='glasso', penalty=0.2)
    assert p_12 > p_11
    assert p_12 > p_22

def test_semi_param_kernel_estimate_warton():
    ssx, test_1, test_2 = make_test_data()
    
    p_10 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0])
    p_11 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='warton', penalty=1)
    assert np.isclose(p_10, p_11)

    p_12 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='warton', penalty=0.8)
    p_22 = pdf_methods.semi_param_kernel_estimate(ssx, test_1[0], shrinkage='warton', penalty=0.8)
    assert p_12 > p_11
    assert p_12 > p_22
