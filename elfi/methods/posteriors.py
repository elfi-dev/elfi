"""The module contains implementations of approximate posteriors."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from elfi.methods.bo.utils import minimize
from elfi.methods.utils import NDimBoundingBox, ModelPrior
from elfi.visualization.visualization import progress_bar
from typing import List, Callable


logger = logging.getLogger(__name__)


# TODO: separate the likelihood to its own class
class BolfiPosterior:
    r"""Container for the approximate posterior in the BOLFI framework.

    Here the likelihood is defined as

    L \propto F((h - \mu) / \sigma)

    where F is the cdf of N(0,1), h is a threshold, and \mu and \sigma are the mean and (noisy)
    standard deviation of the Gaussian process.

    Note that when using a log discrepancy, h should become log(h).

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    """

    def __init__(self, model, threshold=None, prior=None, n_inits=10, max_opt_iters=1000, seed=0):
        """Initialize a BOLFI posterior.

        Parameters
        ----------
        model : elfi.bo.gpy_regression.GPyRegression
            Instance of the surrogate model
        threshold : float, optional
            The threshold value used in the calculation of the posterior, see the BOLFI paper
            for details. By default, the minimum value of discrepancy estimate mean is used.
        prior : ScipyLikeDistribution, optional
            By default uniform distribution within model bounds.
        n_inits : int, optional
            Number of initialization points in internal optimization.
        max_opt_iters : int, optional
            Maximum number of iterations performed in internal optimization.
        seed : int, optional

        """
        super(BolfiPosterior, self).__init__()
        self.threshold = threshold
        self.model = model
        self.random_state = np.random.RandomState(seed)
        self.n_inits = n_inits
        self.max_opt_iters = max_opt_iters

        self.prior = prior
        self.dim = self.model.input_dim

        if self.threshold is None:
            # TODO: the evidence could be used for a good guess for starting locations
            minloc, minval = minimize(
                self.model.predict_mean,
                self.model.bounds,
                grad=self.model.predictive_gradient_mean,
                prior=self.prior,
                n_start_points=self.n_inits,
                maxiter=self.max_opt_iters,
                random_state=self.random_state)
            self.threshold = minval
            logger.info("Using optimized minimum value (%.4f) of the GP discrepancy mean "
                        "function as a threshold" % (self.threshold))

    def rvs(self, size=None, random_state=None):
        """Sample the posterior.

        Currently unimplemented. Please use a sampler to sample from the posterior.
        """
        raise NotImplementedError('Currently not implemented. Please use a sampler to '
                                  'sample from the posterior.')

    def logpdf(self, x):
        """Return the unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float

        """
        return self._unnormalized_loglikelihood(x) + self.prior.logpdf(x)

    def pdf(self, x):
        """Return the unnormalized posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float

        """
        return np.exp(self.logpdf(x))

    def gradient_logpdf(self, x):
        """Return the gradient of the unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        np.array

        """
        grads = self._gradient_unnormalized_loglikelihood(x) + \
            self.prior.gradient_logpdf(x)

        # nan grads are result from -inf logpdf
        # return np.where(np.isnan(grads), 0, grads)[0]
        return grads

    def _unnormalized_loglikelihood(self, x):
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        logpdf = -np.ones(len(x)) * np.inf

        logi = self._within_bounds(x)
        x = x[logi, :]
        if len(x) == 0:
            if ndim == 0 or (ndim == 1 and self.dim > 1):
                logpdf = logpdf[0]
            return logpdf

        mean, var = self.model.predict(x)
        logpdf[logi] = ss.norm.logcdf(self.threshold, mean, np.sqrt(var)).squeeze()

        if ndim == 0 or (ndim == 1 and self.dim > 1):
            logpdf = logpdf[0]

        return logpdf

    def _gradient_unnormalized_loglikelihood(self, x):
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        grad = np.zeros_like(x)

        logi = self._within_bounds(x)
        x = x[logi, :]
        if len(x) == 0:
            if ndim == 0 or (ndim == 1 and self.dim > 1):
                grad = grad[0]
            return grad

        mean, var = self.model.predict(x)
        std = np.sqrt(var)

        grad_mean, grad_var = self.model.predictive_gradients(x)

        factor = -grad_mean * std - (self.threshold - mean) * 0.5 * grad_var / std
        factor = factor / var
        term = (self.threshold - mean) / std
        pdf = ss.norm.pdf(term)
        cdf = ss.norm.cdf(term)

        grad[logi, :] = factor * pdf / cdf

        if ndim == 0 or (ndim == 1 and self.dim > 1):
            grad = grad[0]

        return grad

    # TODO: check if these are used
    def _unnormalized_likelihood(self, x):
        return np.exp(self._unnormalized_loglikelihood(x))

    def _neg_unnormalized_loglikelihood(self, x):
        return -1 * self._unnormalized_loglikelihood(x)

    def _gradient_neg_unnormalized_loglikelihood(self, x):
        return -1 * self._gradient_unnormalized_loglikelihood(x)

    def _neg_unnormalized_logposterior(self, x):
        return -1 * self.logpdf(x)

    def _gradient_neg_unnormalized_logposterior(self, x):
        return -1 * self.gradient_logpdf(x)

    def _within_bounds(self, x):
        x = x.reshape((-1, self.dim))
        logical = np.ones(len(x), dtype=bool)
        for i in range(self.dim):
            logical *= (x[:, i] >= self.model.bounds[i][0])
            logical *= (x[:, i] <= self.model.bounds[i][1])
        return logical

    def plot(self, logpdf=False):
        """Plot the posterior pdf.

        Currently only supports 1 and 2 dimensional cases.

        Parameters
        ----------
        logpdf : bool
            Whether to plot logpdf instead of pdf.

        """
        if logpdf:
            fun = self.logpdf
        else:
            fun = self.pdf

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore')

            if len(self.model.bounds) == 1:
                mn = self.model.bounds[0][0]
                mx = self.model.bounds[0][1]
                dx = (mx - mn) / 200.0
                x = np.arange(mn, mx, dx)
                pd = np.zeros(len(x))
                for i in range(len(x)):
                    pd[i] = fun([x[i]])
                plt.figure()
                plt.plot(x, pd)
                plt.xlim(mn, mx)
                plt.ylim(min(pd) * 1.05, max(pd) * 1.05)
                plt.show()

            elif len(self.model.bounds) == 2:
                x, y = np.meshgrid(
                    np.linspace(*self.model.bounds[0]), np.linspace(*self.model.bounds[1]))
                z = (np.vectorize(lambda a, b: fun(np.array([a, b]))))(x, y)
                plt.contour(x, y, z)
                plt.show()

            else:
                raise NotImplementedError("Currently unsupported for dim > 2")


class RomcPosterior:
    r"""
    Approximation of the Posterior Distribution as defined by the ROMC method

    References
    ----------
    Ikonomov, B., & Gutmann, M. U. (2019). Robust Optimisation Monte Carlo. http://arxiv.org/abs/1904.00670

    """

    def __init__(self,
                 regions: List[NDimBoundingBox],
                 funcs: List[Callable],
                 nuisance: List[int],
                 funcs_unique: List[Callable],
                 prior: ModelPrior,
                 left_lim,
                 right_lim,
                 eps: float):
        """

        Parameters
        ----------
        regions: List of the n-square regions
        funcs: 
        nuisance
        funcs_unique
        prior
        left_lim
        right_lim
        eps
        """
        # assert len(regions) == len(funcs)

        # self.optim_problems = optim_problems
        self.regions = regions
        self.funcs = funcs
        self.nuisance = nuisance
        self.funcs_unique = funcs_unique
        self.prior = prior
        self.eps = eps
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.dim = prior.dim
        self.partition = None

    def _pdf_unnorm_single_point(self, theta: np.ndarray) -> float:
        """

        Parameters
        ----------
        theta: (D,)

        Returns
        -------
        unnormalized pdf evaluation
        """
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 1

        prior = self.prior

        indicator_sum = self._sum_over_indicators(theta)
        # indicator_sum = self._sum_over_regions(theta)

        # prior
        pr = float(prior.pdf(np.expand_dims(theta, 0)))

        val = pr * indicator_sum
        return val

    def _sum_over_indicators(self, theta: np.ndarray) -> int:
        """Computes on how many
        """
        funcs = self.funcs_unique
        eps = self.eps
        nof_inside = 0
        for i in range(len(funcs)):
            func = funcs[i]
            if func(theta) < eps:
                nof_inside += 1
        return nof_inside

    def _sum_over_regions(self, theta: np.ndarray) -> int:
        """Computes on how many
        """
        regions = self.regions

        nof_inside = 0
        for i in range(len(regions)):
            reg = regions[i]
            if reg.contains(theta):
                nof_inside += 1
        return nof_inside

    def _pdf_unnorm(self, theta: np.ndarray):
        """Computes the value of the unnormalized posterior. The operation is NOT vectorized.

        Parameters
        ----------
        theta: np.ndarray (BS, D)

        Returns
        -------
        np.array: (BS,)
        """
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 2
        batch_size = theta.shape[0]

        # iterate over all points
        pdf_eval = []
        for i in range(batch_size):
            pdf_eval.append(self._pdf_unnorm_single_point(theta[i]))
        return np.array(pdf_eval)

    def _approximate_partition(self, nof_points: int = 30):
        """Approximates Z, computing the integral as a sum.

        Parameters
        ----------
        nof_points: int, nof points to use in each dimension
        """
        assert 0 <= self.dim <= 2, "Approximate partition implemented only for 1D, 2D case."
        dim = self.dim
        left_lim = self.left_lim
        right_lim = self.right_lim

        partition = 0
        vol_per_point = np.prod((right_lim - left_lim) / nof_points)

        if dim == 1:
            for i in np.linspace(left_lim[0], right_lim[0], nof_points):
                theta = np.array([[i]])
                partition += self._pdf_unnorm(theta)[0] * vol_per_point
        elif dim == 2:
            for i in np.linspace(left_lim[0], right_lim[0], nof_points):
                for j in np.linspace(left_lim[1], right_lim[1], nof_points):
                    theta = np.array([[i, j]])
                    partition += self._pdf_unnorm(theta)[0] * vol_per_point
        else:
            print("ERROR: Approximate partition is not implemented for D > 2")

        # update inference state
        self.partition = partition
        return partition

    def pdf(self, theta):
        assert theta.ndim == 2
        assert theta.shape[1] == self.dim
        assert self.dim <= 2, "PDF can be computed up to 2 dimensional problems."

        if self.partition is not None:
            partition = self.partition
        else:
            partition = self._approximate_partition()
            self.partition = partition

        pdf_eval = []
        for i in range(theta.shape[0]):
            pdf_eval.append(self._pdf_unnorm(theta[i:i + 1]) / partition)
        return np.array(pdf_eval)

    def sample(self, n2: int) -> (np.ndarray, np.ndarray):
        regions = self.regions
        funcs = self.funcs
        nof_regions = len(regions)
        prior = self.prior
        eps = self.eps

        # loop over all regions and sample
        theta = []
        for i in range(nof_regions):
            theta.append(regions[i].sample(n2))
        theta = np.array(theta)

        # compute weight - o(n1xn2) complexity
        w = []
        distances = []
        for i in range(nof_regions):
            w.append([])
            indicator_region = self.regions[i].contains
            for j in range(n2):
                progress_bar(i*n2 + j, nof_regions*n2, prefix='Progress:', suffix='Complete', length=50)
                cur_theta = theta[i, j]
                q = regions[i].pdf(cur_theta)
                if q == 0.0:
                    print("Zero q")
                # (ii) p
                pr = float(prior.pdf(np.expand_dims(cur_theta, 0)))

                # (iii) indicator
                # # ind = indicator_region(cur_theta)
                # # if not ind:
                # #     print("Negative indicator")
                dist = funcs[i](cur_theta)
                distances.append(dist)
                ind = funcs[i](cur_theta) < eps

                # compute
                if q > 0:
                    res = ind * pr / q
                else:
                    res = 0

                w[i].append(res)

                progress_bar(i * n2 + j + 1, nof_regions * n2, prefix='Progress:', suffix='Complete', length=50)
        w = np.array(w)
        distances = np.array(distances)
        return theta, w, distances

    def compute_expectation(self, h, theta, w):
        h_theta = h(theta)

        numer = np.sum(h_theta * w)
        denom = np.sum(w)
        return numer / denom

    def visualize_region(self, i, eps, samples):
        assert i < len(self.funcs)
        dim = self.dim
        func = self.funcs[i]
        region = self.regions[i]

        if dim == 1:
            plt.figure()
            plt.title("seed = %d" % self.nuisance[i])

            # plot sampled points
            if samples is not None:
                x = samples[i, :, 0]
                plt.plot(x, np.zeros_like(x), "bo", label="samples")

            x = np.linspace(region.center + region.limits[0, 0] - 0.2, region.center + region.limits[0, 1] + 0.2, 30)
            y = [func(np.atleast_1d(theta)) for theta in x]
            plt.plot(x, y, 'r--', label="distance")
            plt.plot(region.center, 0, 'ro', label="center")
            plt.axvspan(region.center + region.limits[0, 0], region.center + region.limits[0, 1])
            plt.axhline(eps, color="g", label="eps")
            plt.legend()
            plt.show(block=False)
        else:
            plt.figure()
            plt.title("seed = %d" % self.nuisance[i])

            max_offset = np.sqrt(2 * (np.max(np.abs(region.limits)) ** 2)) + 0.2
            x = np.linspace(region.center[0] - max_offset, region.center[0] + max_offset, 30)
            y = np.linspace(region.center[1] - max_offset, region.center[1] + max_offset, 30)
            X, Y = np.meshgrid(x, y)

            Z = []
            for k, ii in enumerate(x):
                Z.append([])
                for l, jj in enumerate(y):
                    Z[k].append(func(np.array([X[k, l], Y[k, l]])))
            Z = np.array(Z)
            plt.contourf(X, Y, Z, 100, cmap="RdGy")
            plt.plot(region.center[0], region.center[1], "ro")

            # plot sampled points
            if samples is not None:
                plt.plot(samples[i, :, 0], samples[i, :, 1], "bo", label="samples")

            # plot eigenectors
            x = region.center
            x1 = region.center + region.rotation[:, 0] * region.limits[0][0]
            plt.plot([x[0], x1[0]], [x[1], x1[1]], "y-o", label="-v1, f(-v1)=%.2f" % (func(x1)))
            x3 = region.center + region.rotation[:, 0] * region.limits[0][1]
            plt.plot([x[0], x3[0]], [x[1], x3[1]], "g-o", label="v1, f(v1)=%.2f" % (func(x3)))

            x2 = region.center + region.rotation[:, 1] * region.limits[1][0]
            plt.plot([x[0], x2[0]], [x[1], x2[1]], "k-o", label="-v2, f(-v2)=%.2f" % (func(x2)))
            x4 = region.center + region.rotation[:, 1] * region.limits[1][1]
            plt.plot([x[0], x4[0]], [x[1], x4[1]], "c-o", label="v2, f(v2)=%.2f" % (func(x3)))

            # plot boundaries
            def plot_side(x, x1, x2):
                tmp = x + (x1 - x) + (x2 - x)
                plt.plot([x1[0], tmp[0], x2[0]], [x1[1], tmp[1], x2[1]], "r-o")

            plot_side(x, x1, x2)
            plot_side(x, x2, x3)
            plot_side(x, x3, x4)
            plot_side(x, x4, x1)

            plt.xlabel("th_1")
            plt.ylabel("th_2")

            plt.legend()
            plt.colorbar()
            plt.show(block=False)

