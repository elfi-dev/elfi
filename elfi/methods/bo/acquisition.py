"""Implementations for acquiring locations of new evidence for Bayesian optimization."""

import logging

import numpy as np
import scipy.stats as ss

from elfi.methods.bo.utils import minimize

logger = logging.getLogger(__name__)


class AcquisitionBase:
    """All acquisition functions are assumed to fulfill this interface.

    Gaussian noise ~N(0, self.noise_var) is added to the acquired points. By default,
    noise_var=0. You can define a different variance for the separate dimensions.

    """

    def __init__(self,
                 model,
                 prior=None,
                 n_inits=10,
                 max_opt_iters=1000,
                 noise_var=None,
                 exploration_rate=10,
                 seed=None):
        """Initialize AcquisitionBase.

        Parameters
        ----------
        model : an object with attributes
                    input_dim : int
                    bounds : tuple of length 'input_dim' of tuples (min, max)
                and methods
                    evaluate(x) : function that returns model (mean, var, std)
        prior : scipy-like distribution, optional
            By default uniform distribution within model bounds.
        n_inits : int, optional
            Number of initialization points in internal optimization.
        max_opt_iters : int, optional
            Max iterations to optimize when finding the next point.
        noise_var : float or np.array, optional
            Acquisition noise variance for adding noise to the points near the optimized
            location. If array, must be 1d specifying the variance for different dimensions.
            Default: no added noise.
        exploration_rate : float, optional
            Exploration rate of the acquisition function (if supported)
        seed : int, optional
            Seed for getting consistent acquisition results. Used in getting random
            starting locations in acquisition function optimization.

        """
        self.model = model
        self.prior = prior
        self.n_inits = int(n_inits)
        self.max_opt_iters = int(max_opt_iters)

        if noise_var is not None and np.asanyarray(noise_var).ndim > 1:
            raise ValueError("Noise variance must be a float or 1d vector of variances "
                             "for the different input dimensions.")
        self.noise_var = noise_var
        self.exploration_rate = exploration_rate
        self.random_state = np.random if seed is None else np.random.RandomState(seed)

    def evaluate(self, x, t=None):
        """Evaluate the acquisition function at 'x'.

        Parameters
        ----------
        x : numpy.array
        t : int
            current iteration (starting from 0)

        """
        raise NotImplementedError

    def evaluate_gradient(self, x, t=None):
        """Evaluate the gradient of acquisition function at 'x'.

        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).

        """
        raise NotImplementedError

    def acquire(self, n, t=None):
        """Return the next batch of acquisition points.

        Gaussian noise ~N(0, self.noise_var) is added to the acquired points.

        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acq_batch_index (starting from 0).

        Returns
        -------
        x : np.ndarray
            The shape is (n, input_dim)

        """
        logger.debug('Acquiring the next batch of %d values', n)

        # Optimize the current minimum
        def obj(x):
            return self.evaluate(x, t)

        def grad_obj(x):
            return self.evaluate_gradient(x, t)

        xhat, _ = minimize(
            obj,
            self.model.bounds,
            grad_obj,
            self.prior,
            self.n_inits,
            self.max_opt_iters,
            random_state=self.random_state)

        # Create n copies of the minimum
        x = np.tile(xhat, (n, 1))
        # Add noise for more efficient fitting of GP
        x = self._add_noise(x)

        return x

    def _add_noise(self, x):
        # Add noise for more efficient fitting of GP
        if self.noise_var is not None:
            noise_var = np.asanyarray(self.noise_var)
            if noise_var.ndim == 0:
                noise_var = np.tile(noise_var, self.model.input_dim)

            for i in range(self.model.input_dim):
                std = np.sqrt(noise_var[i])
                if std == 0:
                    continue
                xi = x[:, i]
                a = (self.model.bounds[i][0] - xi) / std
                b = (self.model.bounds[i][1] - xi) / std
                x[:, i] = ss.truncnorm.rvs(
                    a, b, loc=xi, scale=std, size=len(x), random_state=self.random_state)

        return x


class LCBSC(AcquisitionBase):
    r"""Lower Confidence Bound Selection Criterion.

    Srinivas et al. call this GP-LCB.

    LCBSC uses the parameter delta which is here equivalent to 1/exploration_rate.

    Parameter delta should be in (0, 1) for the theoretical results to hold. The
    theoretical upper bound for total regret in Srinivas et al. has a probability greater
    or equal to 1 - delta, so values of delta very close to 1 or over it do not make much
    sense in that respect.

    Delta is roughly the exploitation tendency of the acquisition function.

    References
    ----------
    N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger. Gaussian
    process optimization in the bandit setting: No regret and experimental design. In
    Proc. International Conference on Machine Learning (ICML), 2010

    E. Brochu, V.M. Cora, and N. de Freitas. A tutorial on Bayesian optimization of expensive
    cost functions, with application to active user modeling and hierarchical reinforcement
    learning. arXiv:1012.2599, 2010.

    Notes
    -----
    The formula presented in Brochu (pp. 15) seems to be from Srinivas et al. Theorem 2.
    However, instead of having t**(d/2 + 2) in \beta_t, it seems that the correct form
    would be t**(2d + 2).

    """

    def __init__(self, *args, delta=None, **kwargs):
        """Initialize LCBSC.

        Parameters
        ----------
        args
        delta : float, optional
            In between (0, 1). Default is 1/exploration_rate. If given, overrides the
            exploration_rate.
        kwargs

        """
        if delta is not None:
            if delta <= 0 or delta >= 1:
                logger.warning('Parameter delta should be in the interval (0,1)')
            kwargs['exploration_rate'] = 1 / delta

        super(LCBSC, self).__init__(*args, **kwargs)
        self.name = 'lcbsc'
        self.label_fn = 'Confidence Bound'

    @property
    def delta(self):
        """Return the inverse of exploration rate."""
        return 1 / self.exploration_rate

    def _beta(self, t):
        # Start from 0
        t += 1
        d = self.model.input_dim
        return 2 * np.log(t**(2 * d + 2) * np.pi**2 / (3 * self.delta))

    def evaluate(self, x, t=None):
        r"""Evaluate the Lower confidence bound selection criterion.

        mean - sqrt(\beta_t) * std

        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).

        """
        mean, var = self.model.predict(x, noiseless=True)
        return mean - np.sqrt(self._beta(t) * var)

    def evaluate_gradient(self, x, t=None):
        """Evaluate the gradient of the lower confidence bound selection criterion.

        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).

        """
        mean, var = self.model.predict(x, noiseless=True)
        grad_mean, grad_var = self.model.predictive_gradients(x)

        return grad_mean - 0.5 * grad_var * np.sqrt(self._beta(t) / var)


class MaxVar(AcquisitionBase):
    r"""The maximum variance acquisition method.

    The next evaluation point is acquired in the maximiser of the variance of
    the unnormalised approximate posterior.

    \theta_{t+1} = arg max Var(p(\theta) * p_a(\theta)),

    where the unnormalised likelihood p_a is defined
    using the CDF of normal distribution, \Phi, as follows:

    p_a(\theta) =
        (\Phi((\epsilon - \mu_{1:t}(\theta)) / \sqrt(\sigma_{1:t}(\theta) + \sigma_n))),

    where \epsilon is the ABC threshold, \mu_{1:t} and \sigma_{1:t} are
    determined by the Gaussian process, \sigma_n is the noise.

    References
    ----------
    [1] Järvenpää et al. (2017). arXiv:1704.00520
    [2] Gutmann M U, Corander J (2016). Bayesian Optimization for
    Likelihood-Free Inference of Simulator-Based Statistical Models.
    JMLR 17(125):1−47, 2016. http://jmlr.org/papers/v17/15-017.html

    """

    def __init__(self, quantile_eps=.01, *args, **opts):
        """Initialise MaxVar.

        Parameters
        ----------
        quantile_eps : int, optional
            Quantile of the observed discrepancies used in setting the ABC threshold.

        """
        super(MaxVar, self).__init__(*args, **opts)
        self.name = 'max_var'
        self.label_fn = 'Variance of the Unnormalised Approximate Posterior'
        self.quantile_eps = quantile_eps
        # The ABC threshold is initialised to a pre-set value as the gp is not yet fit.
        self.eps = .1

    def acquire(self, n, t=None):
        """Acquire a batch of acquisition points.

        Parameters
        ----------
        n : int
            Number of acquisitions.
        t : int, optional
            Current iteration, (unused).

        Returns
        -------
        array_like
            Coordinates of the yielded acquisition points.

        """
        logger.debug('Acquiring the next batch of %d values', n)
        gp = self.model

        # Updating the ABC threshold.
        self.eps = np.percentile(gp.Y, self.quantile_eps * 100)

        def _negate_eval(theta):
            return -self.evaluate(theta)

        def _negate_eval_grad(theta):
            return -self.evaluate_gradient(theta)

        # Obtaining the location where the variance is maximised.
        theta_max, _ = minimize(_negate_eval,
                                gp.bounds,
                                _negate_eval_grad,
                                self.prior,
                                self.n_inits,
                                self.max_opt_iters,
                                random_state=self.random_state)

        # Using the same location for all points in theta batch.
        batch_theta = np.tile(theta_max, (n, 1))

        return batch_theta

    def evaluate(self, theta_new, t=None):
        """Evaluate the acquisition function at the location theta_new.

        Parameters
        ----------
        theta_new : array_like
            Evaluation coordinates.
        t : int, optional
            Current iteration, (unused).

        Returns
        -------
        array_like
            Variance of the approximate posterior.

        """
        mean, var = self.model.predict(theta_new, noiseless=True)
        sigma_n = self.model.noise

        # Using the cdf of Skewnorm to avoid explicit Owen's T computation.
        a = np.sqrt(sigma_n) / np.sqrt(sigma_n + 2. * var)  # Skewness.
        scale = np.sqrt(sigma_n + var)
        phi_skew = ss.skewnorm.cdf(self.eps, a, loc=mean, scale=scale)
        phi_norm = ss.norm.cdf(self.eps, loc=mean, scale=scale)
        var_p_a = phi_skew - phi_norm**2

        val_prior = self.prior.pdf(theta_new).ravel()[:, np.newaxis]

        var_approx_posterior = val_prior**2 * var_p_a
        return var_approx_posterior

    def evaluate_gradient(self, theta_new, t=None):
        """Evaluate the acquisition function's gradient at the location theta_new.

        Parameters
        ----------
        theta_new : array_like
            Evaluation coordinates.
        t : int, optional
            Current iteration, (unused).

        Returns
        -------
        array_like
            Gradient of the variance of the approximate posterior

        """
        phi = ss.norm.cdf
        mean, var = self.model.predict(theta_new, noiseless=True)
        grad_mean, grad_var = self.model.predictive_gradients(theta_new)
        sigma_n = self.model.noise
        scale = np.sqrt(sigma_n + var)

        a = (self.eps - mean) / scale
        b = np.sqrt(sigma_n) / np.sqrt(sigma_n + 2 * var)
        grad_a = (-1. / scale) * grad_mean - \
            ((self.eps - mean) / (2. * (sigma_n + var)**(1.5))) * grad_var
        grad_b = (-np.sqrt(sigma_n) / (sigma_n + 2 * var)**(1.5)) * grad_var

        _phi_a = phi(a)
        int_1 = _phi_a - _phi_a**2
        int_2 = phi(self.eps, loc=mean, scale=scale) \
            - ss.skewnorm.cdf(self.eps, b, loc=mean, scale=scale)
        grad_int_1 = (1. - 2 * _phi_a) * \
            (np.exp(-.5 * (a**2)) / np.sqrt(2. * np.pi)) * grad_a
        grad_int_2 = (1. / np.pi) * \
            (((np.exp(-.5 * (a**2) * (1. + b**2))) / (1. + b**2)) * grad_b +
                (np.sqrt(np.pi / 2.) * np.exp(-.5 * (a**2)) * (1. - 2. * phi(a * b)) * grad_a))

        # Obtaining the gradient prior by applying the following rule:
        # (log f(x))' = f'(x)/f(x) => f'(x) = (log f(x))' * f(x)
        term_prior = self.prior.pdf(theta_new).ravel()[:, np.newaxis]
        grad_prior_log = self.prior.gradient_logpdf(theta_new)
        term_grad_prior = term_prior * grad_prior_log

        gradient = 2. * term_prior * (int_1 - int_2) * term_grad_prior + \
            term_prior**2 * (grad_int_1 - grad_int_2)
        return gradient


class UniformAcquisition(AcquisitionBase):
    """Acquisition from uniform distribution."""

    def acquire(self, n, t=None):
        """Return random points from uniform distribution.

        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int, optional
            (unused)

        Returns
        -------
        x : np.ndarray
            The shape is (n, input_dim)

        """
        bounds = np.stack(self.model.bounds)
        return ss.uniform(bounds[:, 0], bounds[:, 1] - bounds[:, 0]) \
            .rvs(size=(n, self.model.input_dim), random_state=self.random_state)
