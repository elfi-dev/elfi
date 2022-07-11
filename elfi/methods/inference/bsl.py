"""This module contains BSL classes."""

__all__ = ['BSL']

import logging
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

import elfi.client
import elfi.visualization.visualization as vis
from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood
from elfi.methods.inference.parameter_inference import ParameterInference
from elfi.methods.results import BslSample
from elfi.methods.utils import arr2d_to_batch, batch_to_arr2d
from elfi.model.elfi_model import ElfiModel, ObservableMixin
from elfi.model.extensions import ModelPrior
from elfi.model.utils import get_summary_names

logger = logging.getLogger(__name__)


class BSL(ParameterInference):
    """Bayesian Synthetic Likelihood for parameter inference.

    For a description of the default BSL see Price et. al. 2018.
    Sampler implemented using Metropolis-Hastings MCMC.

    References
    ----------
    L. F. Price, C. C. Drovandi, A. Lee & D. J. Nott (2018).
    Bayesian Synthetic Likelihood, Journal of Computational and Graphical
    Statistics, 27:1, 1-11, DOI: 10.1080/10618600.2017.1302882

    """

    def __init__(self, model, n_training_data, summary_names=None, sl_method=None,
                 is_misspec=False, **kwargs):
        """Initialize the BSL sampler.

        Parameters
        ----------
        model : ElfiModel
            ELFI graph used by the algorithm.
        n_training_data : int
            Number of simulations for 1 parametric approximation of the likelihood.
        summary_names : str or list, optional
            Summaries for the synthetic likelihood.
        sl_method : callable, optional
            Synthetic likelihood estimation method. Defaults to gaussian_syn_likelihood.
        is_misspec : bool, opt
            Whether misspecified likelihood calculation is used.
        """
        model = self._resolve_model(model)
        self.summary_names = self._resolve_summary_names(model, summary_names)
        output_names = model.parameter_names + self.summary_names
        super(BSL, self).__init__(model, output_names, **kwargs)

        self.random_state = np.random.RandomState(self.seed)

        self.sl_method_fn = sl_method or gaussian_syn_likelihood
        self.is_rbsl = is_misspec

        self.observed = self._get_observed_summary_values()
        self.n_training_data = self._resolve_n_training_data(n_training_data)
        self._ssx = np.zeros((self.n_training_data, self.observed.size))


    def sample(self, n_samples, sigma_proposals, params0=None, param_names=None,
               burn_in=0, logit_transform_bound=None, **kwargs):
        """Sample from the posterior distribution of BSL.

        The specific approximate likelihood estimated depends on the
        BSL class but generally uses a multivariate normal approximation.

        The sampling is performed with a metropolis MCMC sampler

        Parameters
        ----------
        n_samples : int
            Number of requested samples from the posterior. This
            includes burn_in.
        sigma_proposals : np.array of shape (k x k) - k = number of parameters
            Standard deviations for Gaussian proposals of each parameter.
        burn_in : int, optional
            Length of burnin sequence in MCMC sampling. These samples are
            "thrown away". Defaults to 0.
        params0 : Initial values for each sampled parameter.
        logit_transform_bound : np.array of list
            Each list element contains the lower and upper bound for the
            logit transformation of the corresponding parameter.
        param_names : list, optional
            Custom list of parameter names corresponding to the order
            of parameters in params0 and sigma_proposals.

        Returns
        -------
        BslSample

        """
        self.n_samples = n_samples
        self.sigma_proposals = sigma_proposals
        self.burn_in = burn_in
        self.logit_transform_bound = logit_transform_bound

        # allow custom parameter order
        self.param_names = param_names or self.parameter_names
        self.prior = ModelPrior(self.model, parameter_names=self.param_names)

        self._init_sample(n_samples, params0)

        return self.infer(n_samples+1, **kwargs)

    def set_objective(self, n_samples):
        """Set objective for inference."""
        self.objective['n_samples'] = n_samples
        self.objective['n_batches'] = n_samples * int(self.n_training_data / self.batch_size)

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample

        """
        samples_all = dict()
        outputs = dict()

        for ii, p in enumerate(self.param_names):
            samples_all[p] = np.array(self.state['params'][1:, ii])
            outputs[p] = samples_all[p][self.burn_in:]
        if self.is_rbsl:
            outputs['gamma'] = self.state['gammas'][1:]

        acc_rate = self.num_accepted/(self.n_samples - self.burn_in)
        logger.info("MCMC acceptance rate: {}".format(acc_rate))

        return BslSample(
             samples_all=samples_all,  # includes burn_in in samples
             outputs=outputs,
             acc_rate=acc_rate,
             burn_in=self.burn_in,
             parameter_names=self.param_names
        )

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch: dict
            dict with `self.outputs` as keys and the corresponding
            outputs for the batch as values
        batch_index : int

        """
        super(BSL, self).update(batch, batch_index)

        self._merge_batch(batch)
        if self._round_sim == self.n_training_data:
            self._update_sample()
            self._init_round()

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        Parameters
        ----------
        batch_index: int

        Returns
        -------
        batch: dict

        """
        batch_parameters = np.repeat(self._params, self.batch_size, axis=0)
        return arr2d_to_batch(batch_parameters, self.param_names)


    def _init_sample(self, n_samples, params0):

        # check initialisation point
        if params0 is None:
            params0 = self.model.generate(1, self.param_names, seed=self.seed)
            params0 = batch_to_arr2d(params0, self.param_names)
        else:
            if not np.isfinite(self.prior.logpdf(params0)):
                raise ValueError(f'Initial point {params0} does not have support.')

        # initialise sampler state
        self.state['params'] = np.zeros((n_samples + 1, len(self.param_names)))
        self.state['target'] = np.zeros((n_samples + 1))
        self.num_accepted = 0
        self.state['n_samples'] = 0
        self._params = params0
        self._round_sim = 0

        if self.is_rbsl:
            self.state['gammas'] = np.zeros((n_samples + 1, self.observed.size))
            self.rbsl_state = {}
            self.rbsl_state['gamma'] = None
            self.rbsl_state['loglikelihood'] = None
            self.rbsl_state['std'] = None
            self.rbsl_state['sample_mean'] = None
            self.rbsl_state['sample_cov'] = None

    def _init_round(self):

        current = self.state['params'][self.state['n_samples']-1].flatten()
        cov = self.sigma_proposals
        while self.state['n_samples'] < self.n_samples + 1:
            prop = self._propagate_state(mean=current, cov=cov, random_state=self.random_state)
            if np.isfinite(self.prior.logpdf(prop)):
                # start data collection with the proposed parameter values
                self._params = prop
                self._round_sim = 0
                break
            else:
                # reject candidate
                n = self.state['n_samples']
                self.state['params'][n] = self.state['params'][n - 1]
                self.state['target'][n] = self.state['target'][n - 1]
                if self.is_rbsl:
                    self.state['gammas'][n] = self.state['gammas'][n-1]
                self.state['n_samples'] = n + 1

                # update objective
                self.set_objective(self.objective['n_samples'] - 1)

    def _update_sample(self):

        n = self.state['n_samples']

        # 1. estimate synthetic likelihood

        if self.is_rbsl:
            loglikelihood, gamma, loglikelihood_prev = \
                self.sl_method_fn(self._ssx, self.observed, self.rbsl_state)
            self.state['gammas'][n] = gamma
            self.rbsl_state['gamma'] = gamma
            self.rbsl_state['loglikelihood'] = loglikelihood_prev
        else:
            loglikelihood = self.sl_method_fn(self._ssx, self.observed)

        self._report_sample(n, self._params, loglikelihood)

        # 2. update state

        params_current = np.copy(self._params)
        target_current = loglikelihood + self.prior.logpdf(params_current)

        if n == 0:
            accept_candidate = True
        else:
            params_prev = self.state['params'][n - 1]
            if self.is_rbsl:
                target_prev = loglikelihood_prev + self.prior.logpdf(params_prev)
            else:
                target_prev = self.state['target'][n - 1]
            l_ratio = self._get_mh_ratio(params_current, target_current, params_prev, target_prev)
            prob = np.minimum(1.0, l_ratio)
            u = self.random_state.uniform()
            accept_candidate = u < prob

        if accept_candidate:
            self.state['params'][n] = params_current
            self.state['target'][n] = target_current
            if self.is_rbsl:
                self.rbsl_state['loglikelihood'] = loglikelihood
                self.rbsl_state['sample_mean'] = np.mean(self._ssx, axis=0)
                self.rbsl_state['sample_cov'] = np.cov(self._ssx, rowvar=False)
            if n > self.burn_in:
                self.num_accepted += 1
        else:
            # make state same as previous
            self.state['params'][n] = self.state['params'][n - 1]
            self.state['target'][n] = self.state['target'][n - 1]

        self.state['n_samples'] = n + 1

        if hasattr(self, 'burn_in') and n == self.burn_in:
            logger.info("Burn in finished. Sampling...")

    def _report_sample(self, sample_index, params, SL):
        batch_str = "Finished round {}:\n".format(sample_index)
        fill = 6 * ' '
        batch_str += "{}SL: {} at {}\n".format(fill, SL, params)
        logger.debug(batch_str)

    def _propagate_state(self, mean, cov=0.01, random_state=None):
        """Logic for random walk proposal. Returns the proposed parameters.

        Parameters
        ----------
        mean : np.array of floats
        cov : np.array of floats
        random_state : RandomState, optional

        """
        random_state = random_state or np.random
        scipy_randomGen = ss.multivariate_normal
        scipy_randomGen.random_state = random_state
        if self.logit_transform_bound is not None:
            mean_tilde = self._para_logit_transform(mean, self.logit_transform_bound)
            sample = scipy_randomGen.rvs(mean=mean_tilde, cov=cov)
            prop_state = self._para_logit_back_transform(sample, self.logit_transform_bound)
        else:
            prop_state = scipy_randomGen.rvs(mean=mean, cov=cov)

        return np.atleast_2d(prop_state)

    def _get_mh_ratio(self, curr_sample, current, prev_sample, previous):
        """Calculate the Metropolis-Hastings ratio.

        Also transforms the parameter range with logit transform if
        needed.

        Parameters
        ----------
        batch_index: int

        """

        logp2 = 0
        logit_transform_bound = self.logit_transform_bound
        if logit_transform_bound is not None:
            curr_sample = np.array(curr_sample)
            prev_sample = np.array(prev_sample)
            logp2 = self._jacobian_logit_transform(curr_sample,
                                                   logit_transform_bound) - \
                self._jacobian_logit_transform(prev_sample,
                                               logit_transform_bound)
        res = logp2 + current - previous

        # prevent overflow warnings
        res = 700 if res > 700 else res
        res = -700 if res < -700 else res

        return np.exp(res)

    def _para_logit_transform(self, theta, bound):
        """Apply logit transform on the specified theta and bound range.

        Parameters
        ----------
        theta : np.array
            Array of parameter values
        bound: list of np.arrays
            List of bounds for each parameter

        Returns
        -------
        thetaTilde : np.array
            The transformed parameter value array.

        """
        type_bnd = np.matmul(np.isinf(bound), [1, 2])
        type_str = type_bnd.astype(str)
        theta = theta.flatten()
        p = len(theta)
        thetaTilde = np.zeros(p)
        for i in range(p):
            a = bound[i, 0]
            b = bound[i, 1]
            x = theta[i]

            type_i = type_str[i]

            if type_i == '0':
                thetaTilde[i] = np.log((x - a)/(b - x))
            if type_i == '1':
                thetaTilde[i] = np.log(1/(b - x))
            if type_i == '2':
                thetaTilde[i] = np.log(x - a)
            if type_i == '3':
                thetaTilde[i] = x

        return thetaTilde

    def _para_logit_back_transform(self, thetaTilde, bound):
        """Apply back logit transform on the transformed theta values.

        Parameters
        ----------
        thetaTilde : np.array
            Array of parameter values
        bound: list of np.arrays
            List of bounds for each parameter

        Returns
        -------
        theta : np.array
            The transformed parameter value array.

        """
        thetaTilde = thetaTilde.flatten()
        p = len(thetaTilde)
        theta = np.zeros(p)

        type_bnd = np.matmul(np.isinf(bound), [1, 2])
        type_str = type_bnd.astype(str)
        for i in range(p):
            a = bound[i, 0]
            b = bound[i, 1]
            y = thetaTilde[i]
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

    def _jacobian_logit_transform(self, thetaTilde, bound):
        """Find jacobian of logit transform.

        Parameters
        ----------
        thetaTilde : np.array
        bound: list of np.arrays
            List of bounds for each parameter

        Returns
        ----------
        J : np.array
            Jacobian matrix

        """
        type_bnd = np.matmul(np.isinf(bound), [1, 2])
        type_str = type_bnd.astype(str)
        thetaTilde = thetaTilde.flatten()
        p = len(thetaTilde)
        logJ = np.zeros(p)

        for i in range(p):
            y = thetaTilde[i]
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

    # batch acquisition control

    def _resolve_model(self, model):
        """Resolve ELFI model to be used."""
        if not isinstance(model, ElfiModel):
            raise ValueError('model must be an ElfiModel.')
        return model

    def _resolve_n_training_data(self, n_training_data):
        """Resolve the size of training data to be used."""
        if isinstance(n_training_data, int) and n_training_data > 0:
            if n_training_data % self.batch_size == 0:
                return n_training_data
            raise ValueError('n_training_data must be a multiple of batch_size.')
        raise TypeError('n_training_data must be a positive int.')

    def _resolve_summary_names(self, model, summary_names):
        """Resolve summary statistics to be used."""
        if summary_names is None:
            summary_names = get_summary_names(model)
            if len(summary_names) == 0:
                raise NotImplementedError('Could not resolve summary_names based on the model.')
            logger.info('Using all summary statistics in synthetic likelihood estimation.')
            return summary_names
        if isinstance(summary_names, str):
            summary_names = [summary_names]
        if isinstance(summary_names, list):
            if len(summary_names) == 0:
                raise ValueError('summary_names must include at least one item.')
            for summary_name in summary_names:
                if summary_name not in model.nodes:
                    raise ValueError(f'Node \'{summary_name}\' not found in the model.')
                if not isinstance(model[summary_name], ObservableMixin):
                    raise TypeError(f'Node \'{summary_name}\' is not observable.')
            return summary_names
        raise TypeError('summary_names must be a string or a list of strings.')

    def _get_observed_summary_values(self):
        """Get the observed values for summary statistics."""
        obs_ss = [self.model[summary_name].observed for summary_name in self.summary_names]
        return np.column_stack(obs_ss)

    def _new_round(self, batch_index):
        """Check whether batch_index starts a new data collection round."""
        return (batch_index * self.batch_size) % self.n_training_data == 0

    def _merge_batch(self, batch):
        """Add batch to collected data."""
        data = batch_to_arr2d(batch, self.summary_names)
        self._ssx[self._round_sim:self._round_sim + self.batch_size] = data
        self._round_sim += self.batch_size

    def _allow_submit(self, batch_index):
        """Check whether batch_index can be prepared."""
        if self._new_round(batch_index) and self.batches.has_pending:
            return False
        else:
            return super(BSL, self)._allow_submit(batch_index)

