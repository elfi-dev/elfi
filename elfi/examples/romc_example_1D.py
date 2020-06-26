"""Implementation of Experiment 1 of Robust Optimisation Monte Carlo paper."""

import timeit

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.ndimage as ndimage
import scipy.stats as ss

import elfi


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

        c1 = 0.5 + 0.5 ** 4
        c2 = -0.5 + 0.5 ** 4

        tmp_theta = theta[theta <= -0.5]
        samples[theta <= -0.5] = ss.norm(loc=tmp_theta + c1, scale=1).rvs(random_state=seed)
        theta[theta <= -0.5] = np.inf

        tmp_theta = theta[theta <= 0.5]
        samples[theta <= 0.5] = ss.norm(loc=tmp_theta**4, scale=1).rvs(random_state=seed)
        theta[theta <= 0.5] = np.inf

        tmp_theta = theta[theta < np.inf]
        samples[theta < np.inf] = ss.norm(loc=tmp_theta + c2, scale=1).rvs(random_state=seed)
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
        assert data.ndim == 1

        BS = theta.shape[0]
        N = data.shape[0]
        theta = theta.astype(np.float)

        pdf_eval = np.zeros((BS))
        c1 = 0.5 + 0.5 ** 4
        c2 = - 0.5 + 0.5 ** 4

        def help_func(lim, mode):
            tmp_theta = theta[theta <= lim]
            tmp_theta = np.expand_dims(tmp_theta, -1)
            scale = np.ones_like(tmp_theta)
            if mode == 1:
                pdf_eval[theta <= lim] = np.prod(ss.norm(loc=tmp_theta + c1, scale=scale).pdf(x), 1)
            elif mode == 2:
                pdf_eval[theta <= lim] = np.prod(ss.norm(loc=tmp_theta**4, scale=scale).pdf(x), 1)
            elif mode == 3:
                pdf_eval[theta <= lim] = np.prod(ss.norm(loc=tmp_theta + c2, scale=scale).pdf(x), 1)
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


def test_deterministic_functions():
    y = []
    dis = []
    u_list = range(1, 5)
    for u in u_list:

        x = np.linspace(-5., 5., 300)
        y.append([dist.model.generate(batch_size=1, with_values={'theta': np.array([th])}, seed=u)["sim"] for th in x])
        dis.append([dist.model.generate(batch_size=1, with_values={'theta': np.array([th])}, seed=u)["distance"] for th in x])

    plt.figure()
    plt.title("Deterministic Simulator")
    for i in range(len(u_list)):
        plt.plot(x, np.squeeze(y[i]), '--', label="u=%d" % u_list[i])
    plt.ylim(-5, 5)
    plt.ylabel("y = f(theta, u)")
    plt.xlabel("theta")
    plt.legend()
    plt.show(block=False)

    plt.figure()
    plt.title("Distance")
    for i in range(len(u_list)):
        plt.plot(x, np.squeeze(dis[i]), '--', label="u=%d" % u_list[i])
    plt.ylim(-5, 5)
    plt.ylabel("y = f(theta, u)")
    plt.xlabel("theta")
    plt.legend()
    plt.show(block=False)


# Ground-truth part
data = np.array([0.])
a = -2.5  # integration left limit
b = 2.5  # integration right limit

likelihood = Likelihood()
prior = Prior()

# approximate Z
factor = create_factor(x=data)
Z = approximate_Z(factor, a, b)

# Ground-Truth posterior pdf
gt_posterior_pdf = create_gt_posterior(likelihood, prior, data, Z)


############# ELFI PART ################
n1 = 100
n2 = 5
seed = 27
eps = 1
left_lim = np.array([-2.5])
right_lim = np.array([2.5])
dim = data.shape[-1]

# Model Definition
elfi.new_model("1D_example")
elfi_prior = elfi.Prior(Prior(), name="theta")
elfi_simulator = elfi.Simulator(simulate_data, elfi_prior, dim, observed=np.expand_dims(data, 0), name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")

# ROMC
tic = timeit.default_timer()
romc = elfi.ROMC(dist, left_lim=left_lim, right_lim=right_lim)
toc = timeit.default_timer()
print("Time for defining model                          : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
romc.sample_nuisance(n1=n1, seed=seed)
toc = timeit.default_timer()
print("Time for sampling nuisance                       : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
romc.define_optim_problems()
toc = timeit.default_timer()
print("Time for defining optim problems                 : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
romc.solve_optim_problems(seed=seed)
toc = timeit.default_timer()
print("Time for solving optim problems                  : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
romc.filter_solutions(eps)
toc = timeit.default_timer()
print("Time for filtering solutions                     : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
print(romc.estimate_region())
toc = timeit.default_timer()
print("Time for estimating regions                      : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
romc.approximate_partition()
toc = timeit.default_timer()
print("Time for approximating Z                         : %.3f sec \n" % (toc-tic))

print(romc.posterior(np.array([[0]])))
# romc.posterior(np.array([0]))

# Rejection sampling
rej = elfi.Rejection(dist, batch_size=10000, seed=21)
rej_res = rej.sample(n_samples=100, threshold=1)
rejection_posterior_pdf = ss.gaussian_kde(rej_res.samples['theta'].squeeze())

# make plot
plt.figure()
plt.title("Posteriors (Z=%.2f)" % Z)
plt.xlim(-10, 10)
plt.xlabel("theta")
plt.ylabel("Density")
plt.ylim(0, 1)

# plt.hist(rej_res.samples["theta"].squeeze(), 50, density=True)

# plot prior
theta = np.linspace(-10, 10, 1000)
y = prior.pdf(theta)
plt.plot(theta, y, 'b-.', label='Prior')

# plot likelihood
y = likelihood.pdf(x=data, theta=theta)
plt.plot(theta, y, 'r-.', label='Likelihood')

# plot posterior
y = np.squeeze(np.array([gt_posterior_pdf(th) for th in theta]))
plt.plot(theta, y, 'g-.', label="True Posterior")

# # plot rejection posterior
y = rejection_posterior_pdf(theta)
plt.plot(theta, y, '-.', label="Rejection")

# plot ROMC posterior
y = [romc.posterior(np.array([[th]])) for th in theta]
tmp = np.squeeze(np.array(y))
tmp = ndimage.gaussian_filter1d(tmp, 7)
# plt.plot(theta, y, 'y-.', label="ROMC Posterior")
plt.plot(theta, tmp, 'y-.', label="ROMC Posterior")

plt.legend()
plt.show(block=False)

# estimate an expectation
def h(x):
    return x

print(romc.compute_expectation(h, N2=n2, seed=seed+2))
