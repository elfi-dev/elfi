import logging
import numpy as np

import scipy.stats as ss
import matplotlib.pyplot as plt

from elfi.methods.bo.utils import minimize


logger = logging.getLogger(__name__)


# TODO: separate the likelihood to its own class
class BolfiPosterior:
    """
    Container for the approximate posterior in the BOLFI framework, where the likelihood
    is defined as

    L \propto F((h - \mu) / \sigma)

    where F is the cdf of N(0,1), h is a threshold, and \mu and \sigma are the mean and (noisy)
    standard deviation of the Gaussian process.

    Note that when using a log discrepancy, h should become log(h).

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

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
    """

    def __init__(self, model, threshold=None, prior=None, n_inits=10, max_opt_iters=1000,
                 seed=0):
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
            minloc, minval = minimize(self.model.predict_mean,
                                      self.model.bounds,
                                      self.model.predictive_gradient_mean,
                                      self.prior,
                                      self.n_inits,
                                      self.max_opt_iters,
                                      random_state=self.random_state)
            self.threshold = minval
            logger.info("Using optimized minimum value (%.4f) of the GP discrepancy mean "
                        "function as a threshold" % (self.threshold))

    def rvs(self, size=None, random_state=None):
        raise NotImplementedError('Currently not implemented. Please use a sampler to '
                                  'sample from the posterior.')

    def logpdf(self, x):
        """
        Returns the unnormalized log-posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float
        """
        return self._unnormalized_loglikelihood(x) + self.prior.logpdf(x)

    def pdf(self, x):
        """
        Returns the unnormalized posterior pdf at x.

        Parameters
        ----------
        x : np.array

        Returns
        -------
        float
        """
        return np.exp(self.logpdf(x))

    def gradient_logpdf(self, x):
        """
        Returns the gradient of the unnormalized log-posterior pdf at x.

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
        #return np.where(np.isnan(grads), 0, grads)[0]
        return grads

    def _unnormalized_loglikelihood(self, x):
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        logpdf = -np.ones(len(x))*np.inf

        logi = self._within_bounds(x)
        x = x[logi,:]
        if len(x) == 0:
            if ndim == 0 or (ndim==1 and self.dim > 1):
                logpdf = logpdf[0]
            return logpdf

        mean, var = self.model.predict(x)
        logpdf[logi] = ss.norm.logcdf(self.threshold, mean, np.sqrt(var)).squeeze()

        if ndim == 0 or (ndim==1 and self.dim > 1):
            logpdf = logpdf[0]

        return logpdf

    def _gradient_unnormalized_loglikelihood(self, x):
        x = np.asanyarray(x)
        ndim = x.ndim
        x = x.reshape((-1, self.dim))

        grad = np.zeros_like(x)

        logi = self._within_bounds(x)
        x = x[logi,:]
        if len(x) == 0:
            if ndim == 0 or (ndim==1 and self.dim > 1):
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

        if ndim == 0 or (ndim==1 and self.dim > 1):
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
                plt.ylim(min(pd)*1.05, max(pd)*1.05)
                plt.show()

            elif len(self.model.bounds) == 2:
                x, y = np.meshgrid(np.linspace(*self.model.bounds[0]), np.linspace(*self.model.bounds[1]))
                z = (np.vectorize(lambda a,b: fun(np.array([a, b]))))(x, y)
                plt.contour(x, y, z)
                plt.show()

            else:
                raise NotImplementedError("Currently unsupported for dim > 2")
