import numpy as np
import pytest

from elfi.methods import mcmc

# construct a covariance matrix and calculate the precision matrix
n = 5
true_cov = np.random.rand(n, n) * 0.5
true_cov += true_cov.T
true_cov += n * np.eye(n)
prec = np.linalg.inv(true_cov)


# log multivariate Gaussian pdf
def log_pdf(x):
    return -0.5 * x.dot(prec).dot(x)


# gradient of log multivariate Gaussian pdf
def grad_log_pdf(x):
    return -x.dot(prec)


class TestMetropolis():
    def test_metropolis(self):
        n_samples = 200000
        x_init = np.random.rand(n)
        sigma = np.ones(n)
        samples = mcmc.metropolis(n_samples, x_init, log_pdf, sigma)
        assert samples.shape == (n_samples, n)
        cov = np.cov(samples[100000:, :].T)
        assert np.allclose(cov, true_cov, atol=0.3, rtol=0.1)


@pytest.mark.slowtest
class TestNUTS():
    def test_nuts(self):
        n_samples = 100000
        n_adapt = 10000
        x_init = np.random.rand(n)
        samples = mcmc.nuts(n_samples, x_init, log_pdf, grad_log_pdf, n_adapt=n_adapt)
        assert samples.shape == (n_samples, n)
        cov = np.cov(samples[n_adapt:, :].T)
        assert np.allclose(cov, true_cov, atol=0.1, rtol=0.1)


# some data generated in PyStan
chains_Stan = np.array([[0.2955857, 1.27937191, 1.05884099, 0.91236858], [
    0.38128885, 1.34242613, 0.49102573, 0.76061715
], [0.38128885, 1.18404563, 0.49102573,
    0.78910512], [0.38128885, 0.72150199, 0.49102573,
                  1.13845618], [0.38128885, 0.72150199, 0.38102685,
                                0.81298041], [0.26917982, 0.72150199, 0.38102685, 0.81298041],
                        [0.26917982, 0.68149163, 0.45830605,
                         0.86364605], [0.51213898, 0.68149163, 0.29170172, 0.80734373],
                        [0.51213898, 0.85560228, 0.29170172,
                         0.48134129], [0.22711558, 0.85560228, 0.29170172, 0.48134129]]).T

ess_Stan = 4.09
Rhat_Stan = 1.714


def test_ESS():
    assert np.isclose(mcmc.eff_sample_size(chains_Stan), ess_Stan, atol=0.01)


def test_Rhat():
    assert np.isclose(mcmc.gelman_rubin_statistic(chains_Stan), Rhat_Stan, atol=0.01)
