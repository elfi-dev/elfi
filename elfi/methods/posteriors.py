"""The module contains implementations of approximate posteriors."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from elfi.methods.bo.utils import minimize
# from elfi.examples import ma2 as ema2

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

class BslPosterior:
    r"""Container for the approximate posterior in the BSL framework
    """
    def __init__(self, y_obs, model=None, prior=None, seed=0, n_sims=None):
        # print('model', self.model)
        super(BslPosterior, self).__init__()
        self.model = model
        self.random_state = np.random.RandomState(seed)

        self.prior = prior
        self.y_obs = y_obs
        self.n_sims = n_sims if n_sims else 1
        # self.dim = self.model.input_dim

    def logpdf(self, x):
        # def MA2(t1, t2, n_obs=50, batch_size=500, random_state=None):
        #     #Make inputs 2d arrays for numpy  broadcasting with w
        #     # print('t1', type(t1), 't2', type(t2))
        #     t1 = np.asanyarray(t1).reshape((-1, 1))
        #     t2 = np.asanyarray(t1).reshape((-1, 1))
        #     random_state = random_state or np.random
        #     w = random_state.randn(batch_size, n_obs+2) #i.i.d. sequence ~ N(0,1)

        #     # print('t1', t1.shape, 't2', t2.shape, 'w', w.shape)
        #     x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
        #     return x
        
        # print('input', x[0], x[1])
        # sims = MA2(x[0], x[1])
        # sample_mean = sims.mean(0)
        # sample_cov = np.asmatrix(np.cov(np.transpose(sims), bias=False))clea # Compute the estimate SL
        # if np.any(np.isinf(sample_cov)):
        #     return 1
        # print('sample_mean', sample_mean, 'sample_cov', sample_cov.shape)

        # print('sims', sims)
        # m = ema2.get_model(100, true_params=x)

        sim_fn = self.model.get_node('_simulator')['attr_dict']['_operation']
        sum_fn = self.model.get_node('_summary')['attr_dict']['_operation']
        # sim_results = sim_fn(n_obs=self.y_obs.size, batch_size=self.n_sims, *x) # TODO: MAKE AUTOMATIC n_obs, setc
        print('x', x)
        sim_results = sim_fn(x, batch_size=self.n_sims)
        print('sim_results', sim_results.shape)
        print('self.n_sims', self.n_sims)
        print('self.y_obs.size', self.y_obs.size)

        #TODO: HOW ARRANGE SIM RESULTS?

        #TODO: CASE OF NO SUMMARY FUNCTION

        sim_sum = sum_fn(sim_results)
        print('sim_sum', sim_sum)
        # sim_results = sim_results.reshape(self.n_sims, self.y_obs.size)
        # self.y_obs = self.y_obs.reshape(self.y_obs.size)

        sample_mean = sim_sum.mean(0)
        sample_cov = np.asmatrix(np.cov(np.transpose(sim_sum)))
        # sample_mean = sample_mean.reshape(self.y_obs.size, 1)
        print('sample_mean', sample_mean)
        print('sample_cov', sample_cov)
        # print('x', x)
        # print('y_obs', self.y_obs.shape)
        # print('np.array(sample_mean)', sample_mean.shape)
        # print('ample_cov', sample_cov.shape)
        # print(' self.prior.logpdf(x)',  self.prior.logpdf(x))
        # print('sim_results', sim_results[:, :].shape) #only consider 2 dimensions for cov
        res =  ss.multivariate_normal.logpdf(
            self.y_obs,
            mean=sample_mean,
            cov=sample_cov) + self.prior.logpdf(x)
        print('res', res, res.shape)
        return ss.multivariate_normal.logpdf(
            self.y_obs,
            mean=np.array(sample_mean),
            cov=sample_cov) + self.prior.logpdf(x)

    # def _unnormalized_loglikelihood(self, x):
    #     pass



