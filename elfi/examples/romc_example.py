"""Implementation of Experiment 1 of Robust Optimisation Monte Carlo paper"""

import numpy as np
import scipy.stats as ss
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import elfi


class Prior:
    r"""The prior distribution"""

    def rvs(self, size=None, seed=None):
        """

        Parameters
        ----------
        size: np.array or None
        seed: integer or None

        Returns
        -------
        np.array: a sample from the distribution
        """
        return ss.uniform(loc=-2.5, scale=5).rvs(size=size, random_state=seed)

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
        theta: np.array

        Returns
        -------
        np.array (shape: same as theta)

        """
        assert isinstance(theta, np.ndarray)
        theta = theta.astype(np.float)
        samples = np.empty_like(theta)

        c = 0.5 - 0.5 ** 4

        tmp_theta = theta[theta < -0.5]
        samples[theta < -0.5] = ss.norm(loc=tmp_theta + c, scale=1).rvs(random_state=seed)
        theta[theta < -0.5] = np.inf

        tmp_theta = theta[theta < 0.5]
        samples[theta < 0.5] = ss.norm(loc=tmp_theta**4, scale=1).rvs(random_state=seed)
        theta[theta < 0.5] = np.inf

        tmp_theta = theta[theta < np.inf]
        samples[theta < np.inf] = ss.norm(loc=tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta < np.inf] = np.inf

        assert np.allclose(theta, np.inf)
        return samples

    def pdf(self, x, theta):
        """

        Parameters
        ----------
        x: np.array, The observed values. must be broadcastable to theta
        theta: np.array, float, the parameter

        Returns
        -------
        float: the pdf evaluation for specific x, theta
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(theta, np.ndarray)

        # broadcast x to theta shape
        x = np.broadcast_to(x, theta.shape).copy().astype(np.float)

        theta = theta.astype(np.float)

        pdf_eval = np.empty_like(x)
        c = 0.5 - 0.5 ** 4

        def help_func(lim, mode):
            tmp_theta = theta[theta < lim]
            tmp_x = x[theta < lim]
            if mode == 1:
                pdf_eval[theta < lim] = ss.norm(loc=tmp_theta + c, scale=1).pdf(tmp_x)
            elif mode == 2:
                pdf_eval[theta < lim] = ss.norm(loc=tmp_theta**4, scale=1).pdf(tmp_x)
            elif mode == 3:
                pdf_eval[theta < lim] = ss.norm(loc=tmp_theta - c, scale=1).pdf(tmp_x)
            theta[theta < lim] = np.inf
            x[theta < lim] = np.inf

        help_func(lim=-0.5, mode=1)
        help_func(lim=0.5, mode=2)
        help_func(lim=np.inf, mode=3)
        return pdf_eval


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


def simulate_data(theta, batch_size=10000, random_state=None): # , N=100, random_state=None):
    likelihood = Likelihood()
    return likelihood.rvs(theta, seed=random_state)


# # Ground-truth part
data = np.array([0])
a = -2.5  # integration left limit
b = 2.5  # integration right limit

likelihood = Likelihood()
prior = Prior()

# approximate Z
factor = create_factor(x=data)
Z = approximate_Z(factor, a, b)

# Ground-Truth posterior pdf
gt_posterior_pdf = create_gt_posterior(likelihood, prior, data, Z)

# elfi part - define Model
elfi_prior = elfi.Prior("uniform", -2.5, 5, name="theta")
elfi_simulator = elfi.Simulator(simulate_data, elfi_prior, observed=data, name="sim")
dist = elfi.Distance('euclidean', elfi_simulator, name="distance")

# Rejection sampling
rej = elfi.Rejection(dist, batch_size=100000, discrepancy_name="eucl_distance", seed=21)
rej_res = rej.sample(n_samples=10000, threshold=.05)
rejection_posterior_pdf = ss.gaussian_kde(rej_res.samples['theta'])
print(rej_res)

# make plot
plt.figure()
plt.title("Posteriors (Z=%.2f)" % Z)
plt.xlim(-3, 3)
plt.xlabel("theta")
plt.ylabel("Density")
plt.ylim(0, 1)

# plot prior
x = np.linspace(-3, 3, 1000)
y = prior.pdf(x).squeeze()
plt.plot(x, y, 'b-.', label='Prior')

# plot likelihood
y = likelihood.pdf(x=data, theta=x).squeeze()
plt.plot(x, y, 'r-.', label='Likelihood')

# plot posterior
y = np.array([gt_posterior_pdf(x_tmp) for x_tmp in x]).squeeze()
plt.plot(x, y, 'g-.', label="True Posterior")

# plot rejection posterio
y = rejection_posterior_pdf(x)
plt.plot(x, y, '-.', label="Rejection")

plt.legend()
plt.show(block=False)
