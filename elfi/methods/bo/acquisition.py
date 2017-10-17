"""Implementations for acquiring locations of new evidence for Bayesian optimization."""

import logging
import sys

import numpy as np
import scipy.linalg as sl
import scipy.stats as ss

import elfi.methods.mcmc as mcmc
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
        self.seed = 0 if seed is None else seed

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
        (\Phi((\epsilon - \mu_{1:t}(\theta)) / \sqrt(v_{1:t}(\theta) + \sigma2_n))),

    where \epsilon is the ABC threshold, \mu_{1:t} and v_{1:t} are
    determined by the Gaussian process, \sigma2_n is the noise.

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
        sigma2_n = self.model.noise

        # Using the cdf of Skewnorm to avoid explicit Owen's T computation.
        a = np.sqrt(sigma2_n) / np.sqrt(sigma2_n + 2. * var)  # Skewness.
        scale = np.sqrt(sigma2_n + var)
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
        sigma2_n = self.model.noise
        scale = np.sqrt(sigma2_n + var)

        a = (self.eps - mean) / scale
        b = np.sqrt(sigma2_n) / np.sqrt(sigma2_n + 2 * var)
        grad_a = (-1. / scale) * grad_mean - \
            ((self.eps - mean) / (2. * (sigma2_n + var)**(1.5))) * grad_var
        grad_b = (-np.sqrt(sigma2_n) / (sigma2_n + 2 * var)**(1.5)) * grad_var

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


class RandMaxVar(MaxVar):
    r"""The randomised maximum variance acquisition method.

    The next evaluation point is sampled from the density corresponding to the
    variance of the unnormalised approximate posterior (The MaxVar acquisition function).

    \theta_{t+1} ~ q(\theta),

    where q(\theta) \propto Var(p(\theta) * p_a(\theta)) and
    the unnormalised likelihood p_a is defined
    using the CDF of normal distribution, \Phi, as follows:

    p_a(\theta) =
        (\Phi((\epsilon - \mu_{1:t}(\theta)) / \sqrt(\v_{1:t}(\theta) + \sigma2_n))),

    where \epsilon is the ABC threshold, \mu_{1:t} and \v_{1:t} are
    determined by the Gaussian process, \sigma2_n is the noise.


    References
    ----------
    [1] arXiv:1704.00520 (Järvenpää et al., 2017)

    """

    def __init__(self, quantile_eps=.01, sampler='nuts', n_samples=50,
                 limit_faulty_init=10, sigma_proposals_metropolis=None, *args, **opts):
        """Initialise RandMaxVar.

        Parameters
        ----------
        quantile_eps : int, optional
            Quantile of the observed discrepancies used in setting the ABC threshold.
        sampler : string, optional
            Name of the sampler (options: metropolis, nuts).
        n_samples : int, optional
            Length of the sampler's chain for obtaining the acquisitions.
        limit_faulty_init : int, optional
            Limit for the iterations used to obtain the sampler's initial points.
        sigma_proposals_metropolis : array_like, optional
            Standard deviation proposals for tuning the metropolis sampler.
            For the default settings, the sigmas are set to the 1/10
            of the parameter intervals' length.

        """
        super(RandMaxVar, self).__init__(quantile_eps, *args, **opts)
        self.name = 'rand_max_var'
        self.name_sampler = sampler
        self._n_samples = n_samples
        self._limit_faulty_init = limit_faulty_init
        self._sigma_proposals_metropolis = sigma_proposals_metropolis

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
        if n > self._n_samples:
            raise ValueError("The number of acquisitions, n, has to be lower"
                             "than the number of the samples (%d)."
                             .format(self._n_samples))

        logger.debug('Acquiring the next batch of %d values', n)
        gp = self.model

        # Updating the ABC threshold.
        self.eps = np.percentile(gp.Y, self.quantile_eps * 100)

        def _evaluate_gradient_logpdf(theta):
            denominator = self.evaluate(theta)
            if denominator == 0:
                return -np.inf
            pt_eval = self.evaluate_gradient(theta) / denominator
            return pt_eval.ravel()

        def _evaluate_logpdf(theta):
            val_pdf = self.evaluate(theta)
            if val_pdf == 0:
                return -np.inf
            return np.log(val_pdf)

        # Obtaining the RandMaxVar acquisition.
        for i in range(self._limit_faulty_init + 1):
            if i > self._limit_faulty_init:
                raise SystemExit("Unable to find a suitable initial point.")

            # Proposing the initial point.
            theta_init = np.zeros(shape=len(gp.bounds))
            for idx_param, range_bound in enumerate(gp.bounds):
                theta_init[idx_param] = self.random_state.uniform(range_bound[0], range_bound[1])

            # Refusing to accept a faulty initial point.
            if np.isinf(_evaluate_logpdf(theta_init)):
                continue

            # Sampling the acquisition using the chosen sampler.
            if self.name_sampler == 'metropolis':
                if self._sigma_proposals_metropolis is None:
                    # Setting the default values of the sigma proposals to 1/10
                    # of each parameters interval's length.
                    sigma_proposals = []
                    for bound in self.model.bounds:
                        length_interval = bound[1] - bound[0]
                        sigma_proposals.append(length_interval / 10)
                    self._sigma_proposals_metropolis = sigma_proposals
                samples = mcmc.metropolis(self._n_samples,
                                          theta_init,
                                          _evaluate_logpdf,
                                          sigma_proposals=self._sigma_proposals_metropolis,
                                          seed=self.seed)
            elif self.name_sampler == 'nuts':
                samples = mcmc.nuts(self._n_samples,
                                    theta_init,
                                    _evaluate_logpdf,
                                    _evaluate_gradient_logpdf,
                                    seed=self.seed)
            else:
                raise ValueError(
                    "Incompatible sampler. Please check the options in the documentation.")

            # Using the last n points of the MH chain for the acquisition batch.
            batch_theta = samples[-n:, :]
            break

        return batch_theta


class ExpIntVar(MaxVar):
    r"""The Expected Integrated Variance (ExpIntVar) acquisition method.

    The next evaluation point is acquired in the minimiser of
    the expected integrated variance loss function L.

    \theta_{t+1} = arg min_{\theta^* \in \Theta} L_{1:t}(\theta^*), where

    \Theta is the parameter space, and L is approximated using importance sampling as:

    L_{1:t}(\theta^*) \approx 2 * \sum_{i=1}^s (\omega^i * p^2(\theta^i)
                                * w_{1:t+1})(theta^i, \theta^*), where

    \omega^i is an importance weight,
    p^2(\theta^i) is the prior squared, and
    w_{1:t+1})(theta^i, \theta^*) is the future variance of the unnormalised posterior.

    References
    ----------
    [1] arXiv:1704.00520 (Järvenpää et al., 2017)

    """

    def __init__(self, quantile_eps=.01, n_imp_samples=100, iter_imp_resample=2, *args, **opts):
        """Initialise ExpIntVar.

        Parameters
        ----------
        quantile_eps : int, optional
            Quantile of the observed discrepancies used to estimate the discrepancy threshold.
        n_imp_samples : int, optional
            Number of importance samples
        iter_imp_resample : int, optional
            Gap between acquisition iterations in performing importance sampling.

        """
        super(ExpIntVar, self).__init__(quantile_eps, *args, **opts)
        self.name = 'exp_int_var'
        self.label_fn = 'Expected Loss'
        self._n_imp_samples = n_imp_samples
        self._iter_imp_resample = iter_imp_resample

        # Initialising the RandMaxVar density for the importance sampling.
        self.density_is = RandMaxVar(model=self.model,
                                     prior=self.prior,
                                     n_inits=self.n_inits,
                                     seed=self.seed,
                                     quantile_eps=self.quantile_eps)

    def acquire(self, n, t):
        """Acquire a batch of acquisition points.

        Parameters
        ----------
        n : int
            Number of acquisitions.
        t : int
            Current iteration.

        Returns
        -------
        array_like
            Coordinates of the yielded acquisition points.

        """
        logger.debug('Acquiring the next batch of %d values', n)
        gp = self.model

        # Updating the discrepancy threshold.
        self.eps = np.percentile(gp.Y, self.quantile_eps * 100)

        # Performing the importance sampling step every self._iter_imp_resample iterations.
        if t % self._iter_imp_resample == 0:
            self.samples_imp = self.density_is.acquire(self._n_imp_samples)

        # Pre-calculating the omega_imp and prior_imp terms to be used in the evaluate function.
        n_samples = 1
        n_imp, n_dim = self.samples_imp.shape
        self.omegas_imp = np.zeros(shape=(n_samples, n_dim, n_imp))
        self.priors_imp = np.zeros(shape=(n_samples, n_dim, n_imp))
        for idx_is, sample_imp in enumerate(self.samples_imp):
            omega_imp = 1 / self.density_is.evaluate(sample_imp)
            # Suppressing infinite values.
            if omega_imp == np.inf:
                omega_imp = sys.float_info.max
            self.omegas_imp[:, :, idx_is] = omega_imp
            prior_imp = self.prior.pdf(sample_imp)**2
            self.priors_imp[:, :, idx_is] = prior_imp
        self.omegas_imp = self.omegas_imp / np.sum(self.omegas_imp, axis=2)[:, :, np.newaxis]

        self.thetas_old = np.array(gp.X)
        self.sigma_n = gp.noise
        self._K = gp._gp.kern.K
        self.K = self._K(self.thetas_old, self.thetas_old) \
            + self.sigma_n * np.identity(self.thetas_old.shape[0])

        # Obtaining the location where the expected loss is minimised.
        # Note: The gradient is computed numerically as GPy currently does not
        # provide the derivative computations used in Järvenpää et al., 2017.
        theta_min, _ = minimize(self.evaluate,
                                gp.bounds,
                                grad=None,
                                prior=self.prior,
                                n_start_points=self.n_inits,
                                maxiter=self.max_opt_iters,
                                random_state=self.random_state)

        # Using the same location for all points in the batch.
        batch_theta = np.tile(theta_min, (n, 1))
        return batch_theta

    def evaluate(self, theta_new, t=None):
        """Evaluate the acquisition function at the location theta_new.

        The rationale of the ExpIntVar acquisition is based on minimising this
        evaluation function (i.e., minimising the expected loss).

        Parameters
        ----------
        theta_new : array_like
            Evaluation coordinates.
        t : int, optional
            Current iteration, (unused).

        Returns
        -------
        array_like
            Expected loss.

        """
        # Identify the array modalities, shape=(n_samples, n_dim, n_imp):
        # - The batch size (n_samples).
        # - The data dimensionality (n_dim);
        # - The number of importance samples (n_imp);
        # Note: ExpIntVar is based on a maximiser, batch_size > 1 does not improve the performance.
        n_samples = 1
        n_imp, n_dim = self.samples_imp.shape

        # Alter the shape of theta_new.
        if n_dim != 1 and theta_new.ndim == 1:
            theta_new = theta_new[np.newaxis, :]
        elif n_dim == 1 and theta_new.ndim == 1:
            theta_new = theta_new[:, np.newaxis]

        # Prepare the instances for obtaining the integrand term w.
        gp = self.model
        _, var = gp.predict(theta_new, noiseless=True)
        k_old_new = self._K(self.thetas_old, theta_new)
        # Using the Cholesky factorisation to avoid matrix inverse.
        term_chol = sl.cho_solve(sl.cho_factor(self.K), k_old_new)

        # Calculate the integrand term w.
        # Note: below we obtain w's first term as the second does not depend on theta_new;
        # the complete loss function is provided in Järvenpää et al., 2017.
        w = np.zeros(shape=(n_samples, n_dim, n_imp))
        for idx_is, sample_imp in enumerate(self.samples_imp):
            k_imp_new = self._K(sample_imp[np.newaxis, :], theta_new).T
            k_imp_old = self._K(sample_imp[np.newaxis, :], self.thetas_old).T
            cov_imp = k_imp_new - np.dot(k_imp_old.T, term_chol).T
            delta_var_imp = cov_imp**2 / (self.sigma_n + var)

            # Using the cdf of Skewnorm to avoid an explicit Owen's T computation.
            mean_imp, var_imp = gp.predict(sample_imp, noiseless=True)
            a = np.sqrt((self.sigma_n + var_imp - delta_var_imp) /
                        (self.sigma_n + var_imp + delta_var_imp))
            phi_skew_imp = ss.skewnorm.cdf(self.eps, a, loc=mean_imp,
                                           scale=np.sqrt(self.sigma_n + var_imp))
            phi_imp = ss.norm.cdf(self.eps, loc=mean_imp, scale=np.sqrt(self.sigma_n + var_imp))
            T_imp = ((phi_imp - phi_skew_imp) / 2)
            w[:, :, idx_is] = T_imp

        loss = 2 * np.sum(self.omegas_imp * self.priors_imp * w, axis=(1, 2))
        return loss


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
