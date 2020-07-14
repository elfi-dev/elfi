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

        x = []
        for i in range(th1.shape[0]):
            cur_th = np.concatenate((th1[i], th2[i]))
            x.append(ss.multivariate_normal(mean=cur_th, cov=1).rvs(random_state=seed))
        return np.array(x)

    def pdf(self, x, th1, th2):
        """

        Parameters
        ----------
        x: np.array (1xk)
        th1: np.array (1x1)
        th2: np.array (1x1)

        Returns
        -------
        float
        """
        assert isinstance(th1, float)
        assert isinstance(th2, float)
        assert isinstance(x, np.ndarray)

        th = np.stack((th1, th2))
        rv = ss.multivariate_normal(mean=th, cov=1)
        nof_points = x.shape[0]
        prod = 1
        for i in range(nof_points):
            prod *= rv.pdf(x[i])
        return prod


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
    ax.set_title('Ground-Truth Posterior PDF')
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
    
    # Z = ndimage.gaussian_filter(Z, sigma=1)

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


def summarize(x):
    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    return np.prod(x, axis=-1)


data = np.ones((1, 2))
dim = data.shape[-1]
factor = create_factor(data)
Z = approximate_Z(factor)
gt_posterior = create_gt_posterior(factor, Z)


# ELFI part - Model definition
elfi.new_model("2D_example")
elfi_th1 = elfi.Prior(Prior(), name="th1")
elfi_th2 = elfi.Prior(Prior(), name="th2")
elfi_simulator = elfi.Simulator(simulate_data, elfi_th1, elfi_th2, observed=data, name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")
summary = elfi.Summary(summarize, dist, name="summary")


# # ROMC
n1 = 10
n2 = 10
seed = 21
eps = .1
left_lim = np.ones(dim)*-2.5
right_lim = np.ones(dim)*2.5
region_mode = "romc_jacobian"
nof_points = 30

romc = elfi.ROMC(dist, left_lim=left_lim, right_lim=right_lim)
romc.fit_posterior(n1=n1, eps=eps, region_mode=region_mode, seed=seed)
romc.eval_posterior(theta=np.array([[0., 0.]]))
romc.sample(n2=n2)
print("Expected value   : %.3f" % romc.compute_expectation(lambda x: x[:,:,0]+ x[:,:,1]))
romc.visualize_region(1)

# Rejection
rej = elfi.Rejection(summary, batch_size=100, seed=seed)
rej_res = rej.sample(n_samples=100, threshold=eps)
th = np.concatenate((rej_res.samples['th1'], rej_res.samples['th2']), -1)
rejection_posterior_pdf = ss.gaussian_kde(th.T)


# Plots
tic = timeit.default_timer()
plot_gt_posterior(gt_posterior, nof_points=nof_points)
toc = timeit.default_timer()
print("Time for plotting GT posterior                   : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
plot_romc_posterior(romc.eval_posterior, nof_points=nof_points)
toc = timeit.default_timer()
print("Time for plotting ROMC posterior                 : %.3f sec \n" % (toc-tic))

tic = timeit.default_timer()
plot_rejection_posterior(rejection_posterior_pdf, nof_points=nof_points)
toc = timeit.default_timer()
print("Time for plotting ROMC posterior                 : %.3f sec \n" % (toc-tic))


# # TEST for Bounding Box
# th = 237
# theta = np.radians(th)
# c, s = np.cos(theta), np.sin(theta)
# R = np.array(((c, -s), (s, c)))

# center = np.array([-1,-1])
# limits = np.array([[-0.5, 0.5], [-0.5, .5]])

# bb = elfi.methods.utils.NDimBoundingBox(R, center, limits)

# bb.plot(bb.sample(1000))

# np.sum([bb.contains(i) for i in bb.sample(1000)])
