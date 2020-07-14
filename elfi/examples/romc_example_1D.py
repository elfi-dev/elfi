"""Implementation of Experiment 1 of Robust Optimisation Monte Carlo paper."""
import timeit

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.ndimage as ndimage
import scipy.stats as ss
import elfi

np.random.seed(21)

class Prior:
    r"""The prior distribution"""

    def rvs(self, size=None, random_state=None):
        """

        Parameters
        ----------
        size: np.array (BS,) or None (passes automatically from model.generate as (BS,))
        seed: integer or None (passes automatically from model as RandomState object)

        Returns
        -------
        np.array (BSx1)
        """
        # size from (BS,) -> (BS,1)
        if size is not None:
            size = np.concatenate((size, [1]))
        return ss.uniform(loc=-2.5, scale=5).rvs(size=size, random_state=random_state)

    def pdf(self, theta):
        """

        Parameters
        ----------
        theta: np.array or float

        Returns
        -------
        np.array or float (as theta): element-wise pdf evaluation
        """
        return ss.uniform(loc=-2.5, scale=5).pdf(theta)

    def logpdf(self, theta):
        """

        Parameters
        ----------
        theta: np.array or float

        Returns
        -------
        np.array or float (as theta): element-wise pdf evaluation
        """
        return ss.uniform(loc=-2.5, scale=5).logpdf(theta)


class Likelihood:
    r"""Implements the distribution
    P(x|theta) = N(x; theta^4, 1)         if theta in [-0.5, 0.5]
                 N(x; theta - 0.5 + 0.5^4 if theta > 0.5
                 N(x; theta + 0.5 - 0.5^4 if theta < 0.5
    """

    def rvs(self, theta, seed=None):
        """Vectorized sampling from likelihood.

        Parameters
        ----------
        seed: int
        theta: np.array (whichever shape)

        Returns
        -------
        np.array (shape: same as theta)

        """
        assert isinstance(theta, np.ndarray)
        theta = theta.astype(np.float)
        samples = np.empty_like(theta)

        c = 0.5 - 0.5 ** 4

        tmp_theta = theta[theta <= -0.5]
        samples[theta <= -0.5] = ss.norm(loc=-tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta <= -0.5] = np.inf

        tmp_theta = theta[theta <= 0.5]
        samples[theta <= 0.5] = ss.norm(loc=tmp_theta**4, scale=1).rvs(random_state=seed)
        theta[theta <= 0.5] = np.inf

        tmp_theta = theta[theta < np.inf]
        samples[theta < np.inf] = ss.norm(loc=tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta < np.inf] = np.inf

        assert np.allclose(theta, np.inf)
        return samples

    def pdf(self, x, theta):
        """

        Parameters
        ----------
        x: np.array (BS,)
        theta: np.array (BS,)

        Returns
        -------
        np.array: (BS,)
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 1
        assert x.ndim == 1

        BS = theta.shape[0]
        N = x.shape[0]
        theta = theta.astype(np.float)

        pdf_eval = np.zeros((BS))
        c = 0.5 - 0.5 ** 4

        def help_func(lim, mode):
            tmp_theta = theta[theta <= lim]
            tmp_theta = np.expand_dims(tmp_theta, -1)
            scale = np.ones_like(tmp_theta)
            if mode == 1:
                pdf_eval[theta <= lim] = np.prod(ss.norm(loc=-tmp_theta - c, scale=scale).pdf(x), 1)
            elif mode == 2:
                pdf_eval[theta <= lim] = np.prod(ss.norm(loc=tmp_theta**4, scale=scale).pdf(x), 1)
            elif mode == 3:
                pdf_eval[theta <= lim] = np.prod(ss.norm(loc=tmp_theta - c, scale=scale).pdf(x), 1)
            theta[theta <= lim] = np.inf
            # x[theta < lim] = np.inf

        big_M = 10**7
        help_func(lim=-0.5, mode=1)
        help_func(lim=0.5, mode=2)
        help_func(lim=big_M, mode=3)
        assert np.allclose(theta, np.inf)
        return pdf_eval


def summary(x):
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return np.prod(x, 1)


def create_factor(x):
    """Creates the function g(theta) = L(theta)*prior(theta).

    Parameters
    ----------
    x: np.array [1,]: the observation

    Returns
    -------
    function: input theta float, output: g(theta) float
    """
    lik = Likelihood()
    pr = Prior()
    def tmp_func(theta):
        return float(lik.pdf(x, np.array([theta])) * pr.pdf(theta))
    return tmp_func


def approximate_Z(func, a, b):
    """Approximates the partition function with exhaustive integration.

    Parameters
    ----------
    func: callable, integrand
    a: the left limit of the integration
    b: the right limit of the integration

    Returns
    -------
    float: The Z approximation
    """
    return integrate.quad(func, a, b)[0]


def create_gt_posterior(likelihood, prior, data, Z):
    """

    Parameters
    ----------
    likelihood: object of the Likelihood class
    prior: object of the Prior class
    data: np.array, the observed value
    Z: float, Z approximation

    Returns
    -------
    float: evaluation of the posterior at a specific value
    """
    def tmp_func(theta):
        return likelihood.pdf(data, np.array([theta])) * prior.pdf(np.array([theta])) / Z
    return tmp_func


def simulate_data(theta, dim, batch_size=10000, random_state=None):
    likelihood = Likelihood()
    theta = np.repeat(theta, dim, -1)
    return likelihood.rvs(theta, seed=random_state)


# Ground-truth part
data = np.array([1.5])
a = -2.5  # integration left limit
b = 2.5   # integration right limit

likelihood = Likelihood()
prior = Prior()

# approximate Z
factor = create_factor(x=data)
Z = approximate_Z(factor, a, b)

# Ground-Truth posterior pdf
gt_posterior_pdf = create_gt_posterior(likelihood, prior, data, Z)


############# ELFI PART ################
n1 = 30
n2 = 30
seed = 7
eps = .1
quantile = .5
left_lim = np.array([-2.5])
right_lim = np.array([2.5])
dim = data.shape[-1]
region_mode = "gt_full_coverage"
assert region_mode in ["gt_full_coverage", "gt_around_theta", "romc_jacobian"]

# Model Definition
elfi.new_model("1D_example")
elfi_prior = elfi.Prior(Prior(), name="theta")
elfi_simulator = elfi.Simulator(simulate_data, elfi_prior, dim, observed=np.expand_dims(data, 0), name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")

# ROMC sampling
romc = elfi.ROMC(dist, left_lim=left_lim, right_lim=right_lim)
# romc.solve_problems(n1=n1, seed=seed)
# romc.theta_hist()
# romc.estimate_regions(eps=eps, region_mode=region_mode)

romc.fit_posterior(n1=n1, eps=eps, quantile=quantile, region_mode="romc_jacobian", seed=seed)
romc.eval_posterior(theta=np.array([[0.]]))

# romc.theta_hist()
romc.sample(n2=n2)
# print("Expected value   : %.3f" % romc.compute_expectation(lambda x: np.squeeze(x)))
# print("Expected variance: %.3f" % romc.compute_expectation(lambda x: np.squeeze(x)**2))
# romc.visualize_region(0)
# romc.result.plot_marginals(weights=romc.result.weights, bins=200)
# plt.show(block=False)

# Rejection sampling
# rej = elfi.Rejection(dist, batch_size=1000000, seed=21)
# rej_res = rej.sample(n_samples=1000, threshold=eps)
# rejection_posterior_pdf = ss.gaussian_kde(rej_res.samples['theta'].squeeze())

# make plot
plt.figure()
plt.title("Posteriors (Z=%.2f)" % romc.romc_posterior.partition)
plt.xlim(-3, 3)
plt.xlabel("theta")
plt.ylabel("Density")
plt.ylim(0, 1)

# plot prior
theta = np.linspace(-3, 3, 60)
y = prior.pdf(theta)
plt.plot(theta, y, 'b-.', label='Prior')

# plot likelihood
y = likelihood.pdf(x=data, theta=theta)
plt.plot(theta, y, 'r-.', label='Likelihood')

# plot posterior
y = np.squeeze(np.array([gt_posterior_pdf(th) for th in theta]))
plt.plot(theta, y, 'g-.', label="True Posterior")

# # # plot rejection posterior
# # y = rejection_posterior_pdf(theta)
# # plt.plot(theta, y, '-.', label="Rejection")

# plot ROMC posterior
y = [romc.eval_posterior(np.array([[th]])) for th in theta]
tmp = np.squeeze(np.array(y))
plt.plot(theta, tmp, 'y-.', label="ROMC Posterior")

plt.legend()
plt.show(block=False)
