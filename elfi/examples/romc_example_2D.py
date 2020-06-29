"""Simple 2D example illustration."""

import timeit

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.ndimage as ndimage
import scipy.stats as ss
from mpl_toolkits import mplot3d

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
        np.array or float (as theta): element-wise logpdf evaluation
        """
        return ss.uniform(loc=-2.5, scale=5).logpdf(theta)


class Likelihood:
    r"""Implements the distribution
    P(x|theta) = N(x; [th1,th2], sI)
    """

    def rvs(self, th1, th2, seed=None):
        """Vectorized sampling from likelihood.

        Parameters
        ----------
        seed: int
        theta: np.array (whichever shape)

        Returns
        -------
        np.array (shape: same as theta)

        """
        assert isinstance(th1, np.ndarray)
        assert isinstance(th2, np.ndarray)
        assert th1.ndim == 2
        assert th2.ndim == 2
        assert np.allclose(th1.shape, th2.shape)

        th = np.concatenate((th1, th2), axis=-1)
        x = []
        for i in range(th.shape[0]):
            cur_th = th[i, :]
            x.append(ss.multivariate_normal(mean=cur_th, cov=1).rvs(random_state=seed))
        return np.array(x)

    def pdf(self, x, th1, th2):
        """

        Parameters
        ----------
        x: np.array (1x2)
        th1: np.array (1x1)
        th2: np.array (1x1)

        Returns
        -------
        float
        """
        assert isinstance(th1, float)
        assert isinstance(th2, float)
        assert isinstance(x, np.ndarray)
        assert x.shape[0] == 2

        th = np.stack((th1, th2))
        return ss.multivariate_normal(mean=th, cov=1).pdf(x)


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

    def tmp_func(th1, th2):
        return lik.pdf(x, th1, th2) * pr.pdf(th1) * pr.pdf(th2)
    return tmp_func


def approximate_Z(func):
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
    return integrate.dblquad(func, -2.5, 2.5, lambda x: -2.5, lambda x: 2.5)[0]


def create_gt_posterior(factor, Z):
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
    def tmp_func(th1, th2):
        return factor(th1, th2) / Z
    return tmp_func


def plot_gt_posterior(posterior, nof_points):
    plt.figure()
    x = np.linspace(-4, 4, nof_points)
    y = np.linspace(-4, 4, nof_points)

    x, y = np.meshgrid(x, y)

    tmp = []
    for i in range(x.shape[0]):
        tmp.append([])
        for j in range(x.shape[1]):
            tmp[i].append(posterior(x[i, j], y[i, j]))

    z = np.array(tmp)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Posterio PDF')
    plt.xlabel("th_1")
    plt.ylabel("th_2")
    plt.show(block=False)


def plot_romc_posterior(posterior, nof_points):
    plt.figure()
    th1 = np.linspace(-4, 4, nof_points)
    th2 = np.linspace(-4, 4, nof_points)
    X, Y = np.meshgrid(th1, th2)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    th = np.stack((x_flat, y_flat), -1)
    z_flat = posterior(th)
    Z = z_flat.reshape(nof_points, nof_points)
    
    Z = ndimage.gaussian_filter(Z, sigma=1)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('ROMC Posterior PDF')
    plt.xlabel("th_1")
    plt.ylabel("th_2")
    plt.show(block=False)


def plot_rejection_posterior(posterior, nof_points):
    plt.figure()
    th1 = np.linspace(-4, 4, nof_points)
    th2 = np.linspace(-4, 4, nof_points)
    X, Y = np.meshgrid(th1, th2)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    th = np.stack((x_flat, y_flat), -1)
    z_flat = posterior(th.T)
    Z = z_flat.reshape(nof_points, nof_points)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Rejection Posterior PDF')
    plt.xlabel("th_1")
    plt.ylabel("th_2")
    plt.show(block=False)


def simulate_data(th1, th2, batch_size=10000, random_state=None):
    likelihood = Likelihood()
    return likelihood.rvs(th1, th2, seed=random_state)


# def prior_flat(th):
#     assert th.ndim == 2

#     pr = Prior()
#     th0 = pr.pdf(th[:, 0])
#     th1 = pr.pdf(th[:, 1])

#     return th0*th1


data = np.array([[2., 1.]])

factor = create_factor(data[0])
Z = approximate_Z(factor)
gt_posterior = create_gt_posterior(factor, Z)


# ELFI part - Model definition
elfi.new_model("2D_example")
elfi_th1 = elfi.Prior(Prior(), name="th1")
elfi_th2 = elfi.Prior(Prior(), name="th2")
elfi_simulator = elfi.Simulator(simulate_data, elfi_th1, elfi_th2, observed=data[0], name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")

# ROMC
n1 = 100
n2 = 200
seed = 21
eps = .75
dim = data.shape[-1]
left_lim = np.ones(dim)*-2.5
right_lim = np.ones(dim)*2.5
nof_points = 30


tic = timeit.default_timer()
romc = elfi.ROMC(dist, left_lim, right_lim)
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
print(romc.eval_unnorm_post(np.array([[0, 0]])))
toc = timeit.default_timer()
print("Time for evaluating unnormalized_posterior       : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
romc.approximate_partition(nof_points=nof_points)
toc = timeit.default_timer()
print("Time for approximating partition value           : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
print(romc.posterior(np.array([[0, 0]])))
toc = timeit.default_timer()
print("Time for evaluating posterior at single point    : %.3f sec \n" % (toc-tic))


def h(x):
    return np.sum(x, axis=1)


tic = timeit.default_timer()
print(romc.compute_expectation(h, N2=n2, seed=seed+2))
toc = timeit.default_timer()
print("Time for computing expectation                   : %.3f sec \n" % (toc-tic))


# Rejection
rej = elfi.Rejection(dist, batch_size=10000, seed=seed)
rej_res = rej.sample(n_samples=100, threshold=.3)
th = np.concatenate((rej_res.samples['th1'], rej_res.samples['th2']), -1)
rejection_posterior_pdf = ss.gaussian_kde(th.T)


# Plots
tic = timeit.default_timer()
plot_gt_posterior(gt_posterior, nof_points=nof_points)
toc = timeit.default_timer()
print("Time for plotting GT posterior                   : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
plot_romc_posterior(romc.posterior, nof_points=nof_points)
toc = timeit.default_timer()
print("Time for plotting ROMC posterior                 : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
plot_rejection_posterior(rejection_posterior_pdf, nof_points=nof_points)
toc = timeit.default_timer()
print("Time for plotting ROMC posterior                 : %.3f sec \n" % (toc-tic))
