"""This module contains BSL classes."""

__all__ = ['BSL']

import logging
from functools import partial

import numpy as np

from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood
from elfi.methods.bsl.slice_gamma_mean import slice_gamma_mean
from elfi.methods.bsl.slice_gamma_variance import slice_gamma_variance
from elfi.methods.inference.parameter_inference import ModelBased
from elfi.methods.results import BslSample
from elfi.methods.utils import batch_to_arr2d
from elfi.model.extensions import ModelPrior

logger = logging.getLogger(__name__)


class BSL(ModelBased):
    """Bayesian Synthetic Likelihood for parameter inference.

    For a description of the default BSL see Price et. al. 2018.
    Sampler implemented using Metropolis-Hastings MCMC.

    References
    ----------
    L. F. Price, C. C. Drovandi, A. Lee & D. J. Nott (2018).
    Bayesian Synthetic Likelihood, Journal of Computational and Graphical
    Statistics, 27:1, 1-11, DOI: 10.1080/10618600.2017.1302882

    """

    def __init__(self, model, n_sim_round, feature_names=None, likelihood=None, **kwargs):
        """Initialize the BSL sampler.

        Parameters
        ----------
        model : ElfiModel
            ELFI graph used by the algorithm.
        n_sim_round : int
            Number of simulations for 1 parametric approximation of the likelihood.
        feature_names : str or list, optional
            Features used in synthetic likelihood estimation. Defaults to all summary statistics.
        likelihood : callable, optional
            Synthetic likelihood estimation method. Defaults to gaussian_syn_likelihood.

        """
        super().__init__(model, n_sim_round, feature_names=feature_names, **kwargs)
        self.random_state = np.random.RandomState(self.seed)

        self.likelihood = likelihood or gaussian_syn_likelihood
        self.is_misspec = isinstance(likelihood, partial) and 'adjustment' in likelihood.keywords

        self.param_names = None
        self.prior = None
        self.sigma_proposals = None
        self.burn_in = 0
        self.logit_transform_bound = None
        self.gamma_sampler = None
        self.gamma_sampler_state = {}

    @property
    def parameter_names(self):
        """Return the parameters to be inferred."""
        return self.param_names or self.model.parameter_names

    def sample(self, n_samples, sigma_proposals, params0=None, param_names=None,
               burn_in=0, logit_transform_bound=None, tau=0.5, w=1, max_iter=1000,
               **kwargs):
        """Sample from the posterior distribution of BSL.

        The specific approximate likelihood estimated depends on the
        BSL class but generally uses a multivariate normal approximation.

        The sampling is performed with a metropolis MCMC sampler, and gamma parameters
        are sampled with a slice sampler when adjustment for model misspecification is
        used.

        Parameters
        ----------
        n_samples : int
            Number of requested samples from the posterior. This includes burn_in.
        sigma_proposals : np.array of shape (k x k) - k = number of parameters
            Standard deviations for Gaussian proposals of each parameter.
        params0 : array_like, optional
            Initial values for each sampled parameter.
        param_names : list, optional
            Custom list of parameter names corresponding to the order
            of parameters in params0 and sigma_proposals.
        burn_in : int, optional
            Length of burnin sequence in MCMC sampling. These samples are
            "thrown away". Defaults to 0.
        logit_transform_bound : list, optional
            Each list element contains the lower and upper bound for the
            logit transformation of the corresponding parameter.
        tau : float, optional
            Scale parameter for the prior distribution used by the gamma sampler.
        w : float, optional
            Step size used by the gamma sampler.
        max_iter : int, optional
            Maximum number of iterations used by the gamma sampler.

        Returns
        -------
        BslSample

        """
        self.sigma_proposals = sigma_proposals
        self.param_names = param_names
        self.prior = ModelPrior(self.model, parameter_names=self.parameter_names)
        self.burn_in = burn_in
        if logit_transform_bound is not None:
            self.logit_transform_bound = np.array(logit_transform_bound)
        else:
            self.logit_transform_bound = None
        if self.is_misspec:
            self.gamma_sampler, gamma0 = self._resolve_gamma_sampler(tau, w, max_iter)
        else:
            gamma0 = None

        self._init_state(n_samples, params0, gamma0)
        return self.infer(n_samples, **kwargs)

    def _resolve_gamma_sampler(self, tau, w, max_iter):
        """Return sampler and initial value for gamma parameters."""
        # resolve sampler
        sampler = {'mean': slice_gamma_mean, 'variance': slice_gamma_variance}
        sampler = sampler[self.likelihood.keywords['adjustment']]
        sampler = partial(sampler, tau=tau, w=w, max_iter=max_iter, random_state=self.random_state)

        # resolve initial value
        gamma0 = {'mean': 0.0, 'variance': tau}
        gamma0 = np.repeat(gamma0[self.likelihood.keywords['adjustment']], self.observed.size)

        return sampler, gamma0

    def _init_state(self, n_samples, params0=None, gamma0=None):
        """Initialise method state."""
        super()._init_state()

        # check initialisation point
        if params0 is None:
            params0 = self.model.generate(1, self.parameter_names, seed=self.seed)
            params0 = batch_to_arr2d(params0, self.parameter_names)
        else:
            params0 = np.array(params0)
            if not np.isfinite(self.prior.logpdf(params0)):
                raise ValueError('Initial point {} is outside prior support.'.format(params0))

        # initialise sampler state
        self.state['n_samples'] = 0
        self.num_accepted = 0
        self.state['params'] = np.zeros((n_samples, len(self.parameter_names)))
        self.state['params'][0] = params0
        self.state['logprior'] = np.zeros((n_samples))
        self.state['logprior'][0] = self.prior.logpdf(params0)
        self.state['logposterior'] = np.zeros((n_samples))
        if self.is_misspec:
            self.state['gamma'] = np.zeros((n_samples, self.observed.size))
            self.state['gamma'][0] = gamma0
            self.gamma_sampler_state = {'gamma': gamma0}

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : BslSample

        """
        samples_all = dict()

        for ii, p in enumerate(self.parameter_names):
            samples_all[p] = np.array(self.state['params'][:, ii])
        if self.is_misspec:
            samples_all['gamma'] = self.state['gamma'][:]

        acc_rate = self.num_accepted/(self.state['n_samples'] - self.burn_in)
        logger.info("MCMC acceptance rate: {}".format(acc_rate))

        return BslSample(
            method_name='BSL',
            samples_all=samples_all,  # includes burn_in in samples
            acc_rate=acc_rate,
            burn_in=self.burn_in,
            n_sim=self.state['n_sim'],
            parameter_names=self.parameter_names
        )

    @property
    def current_params(self):
        """Return parameter values explored in the current round.

        BSL runs simulations with the candidate parameter values stored in method state.

        """
        return self.state['params'][self.state['n_samples']]

    def _init_round(self):
        """Initialise a new data collection round.

        BSL samples new candidate parameters from a proposal distribution.

        """
        while self.state['n_samples'] < len(self.state['params']):
            n = self.state['n_samples']
            if self.is_misspec:
                gamma, ll = self.gamma_sampler(self.observed, **self.gamma_sampler_state)
                self.gamma_sampler_state['gamma'] = gamma
                self.gamma_sampler_state['loglik'] = ll
                self.state['gamma'][n] = gamma
                self.state['logposterior'][n-1] = ll + self.state['logprior'][n-1]
            # sample candidate parameter values
            prop = self._propagate_state()
            logprior = self.prior.logpdf(prop)
            if np.isfinite(logprior):
                # start data collection with the proposed parameter values
                self.state['logprior'][n] = logprior
                self.state['params'][n] = prop
                self.state['n_sim_round'] = 0
                break
            else:
                # reject candidate
                self.state['logprior'][n] = self.state['logprior'][n-1]
                self.state['params'][n] = self.state['params'][n-1]
                self.state['logposterior'][n] = self.state['logposterior'][n-1]
                self.state['n_samples'] += 1
                # update inference objective
                self.set_objective(self.objective['round'] - 1)

    def _process_simulated(self):
        """Process the simulated data.

        BSL uses the simulated data to calculate an acceptance probability for the candidate
        parameters. The acceptance probability is calculated based on a synthetic likelihood
        score estimated based on the observed and simulated data.

        """
        # estimate synthetic likelihood
        if not np.all(np.isfinite(self.simulated)):
            loglikelihood = np.NINF
        else:
            if self.is_misspec:
                gamma = self.gamma_sampler_state['gamma']
                loglikelihood = self.likelihood(self.simulated, self.observed, gamma=gamma)
            else:
                loglikelihood = self.likelihood(self.simulated, self.observed)

        n = self.state['n_samples']
        if not np.isfinite(loglikelihood):
            if n == 0:
                raise RuntimeError('Estimated likelihood not finite on initialisation round.')
            logger.warning('Estimated likelihood not finite.')
        logger.debug('SL {} at {}'.format(loglikelihood, self.current_params))

        # update state
        self.state['logposterior'][n] = loglikelihood + self.state['logprior'][n]

        if n == 0:
            accept_candidate = True
        else:
            l_ratio = self._get_mh_ratio()
            prob = np.minimum(1.0, l_ratio)
            u = self.random_state.uniform()
            accept_candidate = u < prob

        if accept_candidate:
            if self.is_misspec:
                # update gamma sampler state
                self.gamma_sampler_state['loglik'] = loglikelihood
                self.gamma_sampler_state['sample_mean'] = np.mean(self.simulated, axis=0)
                self.gamma_sampler_state['sample_cov'] = np.cov(self.simulated, rowvar=False)
            if n >= self.burn_in:
                self.num_accepted += 1
        else:
            # make state same as previous
            self.state['logprior'][n] = self.state['logprior'][n - 1]
            self.state['params'][n] = self.state['params'][n - 1]
            self.state['logposterior'][n] = self.state['logposterior'][n - 1]
        self.state['n_samples'] += 1

        if self.state['n_samples'] == self.burn_in:
            logger.info("Burn in finished. Sampling...")

    def _propagate_state(self):
        """Generate random walk proposal."""
        mean = self.state['params'][self.state['n_samples'] - 1]
        if self.logit_transform_bound is not None:
            mean_tilde = self._para_logit_transform(mean, self.logit_transform_bound)
            sample = self.random_state.multivariate_normal(mean_tilde, self.sigma_proposals)
            prop_state = self._para_logit_back_transform(sample, self.logit_transform_bound)
        else:
            prop_state = self.random_state.multivariate_normal(mean, self.sigma_proposals)

        return np.atleast_2d(prop_state)

    def _get_mh_ratio(self):
        """Calculate the Metropolis-Hastings ratio.

        Takes into account the transformed parameter range if needed.

        """
        n = self.state['n_samples']
        current = self.state['logposterior'][n]
        previous = self.state['logposterior'][n-1]
        logp2 = 0
        if self.logit_transform_bound is not None:
            curr_sample = self.state['params'][n]
            prev_sample = self.state['params'][n-1]
            logp2 = self._jacobian_logit_transform(curr_sample, self.logit_transform_bound) - \
                self._jacobian_logit_transform(prev_sample, self.logit_transform_bound)
        res = logp2 + current - previous

        # prevent overflow warnings
        res = 700 if res > 700 else res
        res = -700 if res < -700 else res

        return np.exp(res)

    @staticmethod
    def _para_logit_transform(theta, bound):
        """Apply logit transform on the specified theta and bound range.

        Parameters
        ----------
        theta : np.array
            Array of parameter values
        bound: np.array
            Bounds for each parameter

        Returns
        -------
        theta_tilde : np.array
            The transformed parameter value array.

        """
        type_bnd = np.matmul(np.isinf(bound), [1, 2])
        type_str = type_bnd.astype(str)
        theta = theta.flatten()
        p = len(theta)
        theta_tilde = np.zeros(p)
        for i in range(p):
            a = bound[i, 0]
            b = bound[i, 1]
            x = theta[i]

            type_i = type_str[i]

            if type_i == '0':
                theta_tilde[i] = np.log((x - a)/(b - x))
            if type_i == '1':
                theta_tilde[i] = np.log(1/(b - x))
            if type_i == '2':
                theta_tilde[i] = np.log(x - a)
            if type_i == '3':
                theta_tilde[i] = x

        return theta_tilde

    @staticmethod
    def _para_logit_back_transform(theta_tilde, bound):
        """Apply back logit transform on the transformed theta values.

        Parameters
        ----------
        theta_tilde : np.array
            Array of parameter values
        bound: np.array
            Bounds for each parameter

        Returns
        -------
        theta : np.array
            The transformed parameter value array.

        """
        theta_tilde = theta_tilde.flatten()
        p = len(theta_tilde)
        theta = np.zeros(p)

        type_bnd = np.matmul(np.isinf(bound), [1, 2])
        type_str = type_bnd.astype(str)
        for i in range(p):
            a = bound[i, 0]
            b = bound[i, 1]
            y = theta_tilde[i]
            ey = np.exp(y)
            type_i = type_str[i]

            if type_i == '0':
                theta[i] = a/(1 + ey) + b/(1 + (1/ey))
            if type_i == '1':
                theta[i] = b-(1/ey)
            if type_i == '2':
                theta[i] = a + ey
            if type_i == '3':
                theta[i] = y

        return theta

    @staticmethod
    def _jacobian_logit_transform(theta_tilde, bound):
        """Find Jacobian of logit transform.

        Parameters
        ----------
        theta_tilde : np.array
            Array of parameter values
        bound: np.array
            Bounds for each parameter

        Returns
        ----------
        J : np.array
            Jacobian matrix

        """
        type_bnd = np.matmul(np.isinf(bound), [1, 2])
        type_str = type_bnd.astype(str)
        theta_tilde = theta_tilde.flatten()
        p = len(theta_tilde)
        logJ = np.zeros(p)

        for i in range(p):
            y = theta_tilde[i]
            type_i = type_str[i]
            if type_i == '0':
                a = bound[i, 0]
                b = bound[i, 1]
                ey = np.exp(y)
                logJ[i] = np.log(b-a) - np.log((1/ey) + 2 + ey)

            if type_i == '1':
                logJ[i] = y
            if type_i == '2':
                logJ[i] = y
            if type_i == '3':
                logJ[i] = 0
        J = np.sum(logJ)
        return J
