"""This module contains BSL classes."""

__all__ = ['BSL']

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

import elfi.client
import elfi.visualization.visualization as vis
from elfi.methods.inference.samplers import Sampler
from elfi.methods.results import BslSample
from elfi.methods.utils import arr2d_to_batch
from elfi.model.extensions import ModelPrior

logger = logging.getLogger(__name__)


class BSL(Sampler):
    """Bayesian Synthetic Likelihood for parameter inference.

    For a description of the default BSL see Price et. al. 2018.
    Sampler implemented using Metropolis-Hastings MCMC.

    References
    ----------
    L. F. Price, C. C. Drovandi, A. Lee & D. J. Nott (2018).
    Bayesian Synthetic Likelihood, Journal of Computational and Graphical
    Statistics, 27:1, 1-11, DOI: 10.1080/10618600.2017.1302882

    """

    def __init__(self, model, batch_size, bsl_name=None,
                 observed=None, output_names=None, seed=None, **kwargs):
        """Initialize the BSL sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        bsl_name : str, optional
            Specify which node to target if model is an ElfiModel.
        observed : np.array, optional
            If not given defaults to observed generated in model.
        output_names : list
            Additional outputs from the model to be included in the inference
            result, e.g. corresponding summaries to the acquired samples
        batch_size : int, optional
            The number of parameter evaluations in each pass through the
            ELFI graph. When using a vectorized simulator, using a suitably
            large batch_size can provide a significant performance boost.
            In the context of BSL, this is the number of simulations for 1
            parametric approximation of the likelihood.
        seed : int, optional
            Seed for the data generation from the ElfiModel

        """
        model, bsl_name = self._resolve_model(model, bsl_name)
        if output_names is None:
            output_names = []
        summary_names = [summary.name for summary in
                         model[bsl_name].parents]
        self.summary_names = summary_names

        output_summaries = [name for name in summary_names if name in
                            output_names]

        output_names = output_names + summary_names + model.parameter_names + \
            [bsl_name]
        self.bsl_name = bsl_name
        super(BSL, self).__init__(
            model, output_names, batch_size, seed, **kwargs)

        self.random_state = np.random.RandomState(self.seed)

        if isinstance(summary_names, str):
            self.summary_names = np.array([summary_names])

        self.observed = observed
        self.prior_state = dict()
        self.prop_state = dict()
        self._prior = ModelPrior(model)
        self.num_accepted = 0
        self.output_summaries = output_summaries
        self.param_names = None  # set in sampling function.

        for name in self.output_names:
            if name not in self.parameter_names:
                self.state[name] = list()

        if self.observed is None:
            self.observed = self._get_observed_summary_values()

    def _get_observed_summary_values(self):
        """Get the observed values for summary statistics.

        Returns:
        obs_ss : np.ndarray
            The observed summary statistic vector.

        """
        obs_ss = [self.model[summary_name].observed for summary_name in
                  self.summary_names]
        obs_ss = np.concatenate(obs_ss)
        return obs_ss

    def sample(self, n_samples, burn_in=0, params0=None, sigma_proposals=None,
               logit_transform_bound=None, param_names=None,
               **kwargs):
        """Sample from the posterior distribution of BSL.

        The specific approximate likelihood estimated depends on the
        BSL class but generally uses a multivariate normal approximation.

        The sampling is performed with a metropolis MCMC sampler

        Parameters
        ----------
        n_samples : int
            Number of requested samples from the posterior. This
            includes burn_in.
        burn_in : int, optional
            Length of burnin sequence in MCMC sampling. These samples are
            "thrown away". Defaults to 0.
        params0 : Initial values for each sampled parameter.
        sigma_proposals : np.array of shape (k x k) - k = number of parameters
            Standard deviations for Gaussian proposals of each parameter.
        logit_transform_bound : np.array of list
            Each list element contains the lower and upper bound for the
            logit transformation of the corresponding parameter.
        param_names : list, optional
            Custom list of parameter names corresponding to the order
            of parameters in params0 and sigma_proposals. The default
j        Returns
        -------
        BslSample

        """
        self.params0 = params0
        self.sigma_proposals = sigma_proposals
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.logit_transform_bound = logit_transform_bound
        self.state['logposterior'] = np.empty(self.n_samples)
        self.state['logprior'] = np.empty(self.n_samples)

        self.param_names = param_names

        for parameter_name in self.parameter_names:
            self.state[parameter_name] = np.empty(self.n_samples)

        return super().sample(n_samples, **kwargs)

    def set_objective(self, *args, **kwargs):
        """Set objective for inference."""
        self.objective['batch_size'] = self.batch_size
        if hasattr(self, 'n_samples'):
            self.objective['n_batches'] = self.n_samples
            self.objective['n_sim'] = self.n_samples * self.batch_size

    def _extract_result_kwargs(self):
        kwargs = super(Sampler, self)._extract_result_kwargs()
        # add in option to use custom parameter name order
        parameter_names = kwargs['parameter_names']
        if self.param_names is not None:
            parameter_names = self.param_names
        kwargs['parameter_names'] = parameter_names
        return kwargs

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample

        """
        outputs = dict()
        samples_all = dict()

        summary_delete = [name for name in self.summary_names if name not in
                          self.output_summaries]

        for p in self.output_names:
            if p in summary_delete:
                continue
            sample = [state_p for ii, state_p in enumerate(self.state[p])]
            samples_all[p] = np.array(sample)
            output = [state_p for ii, state_p in enumerate(self.state[p])
                      if ii >= self.burn_in]
            outputs[p] = np.array(output)

        acc_rate = self.num_accepted/(self.n_samples - self.burn_in)
        logger.info("MCMC acceptance rate: {}".format(acc_rate))

        return BslSample(
             samples_all=samples_all,  # includes burn_in in samples
             outputs=outputs,
             acc_rate=acc_rate,
             burn_in=self.burn_in,
             **self._extract_result_kwargs()
        )

    def _report_batch(self, batch_index, params, SL):
        batch_str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        batch_str += "{}SL: {} at {}\n".format(fill, SL, params)
        logger.debug(batch_str)

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

        ssx = np.column_stack([batch[name] for name in self.summary_names])

        dim1, dim2 = ssx.shape[0:2]
        ssx = ssx.reshape((dim1, dim2))

        # allow option to use custom param names
        param_names = self.param_names if self.param_names \
            is not None else self.parameter_names

        prior_vals = [batch[p][0] for p in param_names]
        prior_vals = np.array(prior_vals)
        self.state['logprior'][batch_index] = self._prior.logpdf(prior_vals)
        self.state['logposterior'][batch_index] = \
            batch[self.bsl_name] + self.state['logprior'][batch_index]
        for s in self.output_names:
            if s not in param_names:
                self.state[s].append(batch[s])

        tmp = {p: self.prop_state[0, i] for i, p in
               enumerate(param_names)}

        for p in param_names:
            self.state[p][batch_index] = tmp[p]

        if hasattr(self, 'burn_in') and batch_index == self.burn_in:
            logger.info("Burn in finished. Sampling...")

        if not self.start_new_chain:
            l_ratio = self._get_mh_ratio(batch_index)
            prob = np.minimum(1.0, l_ratio)
            u = self.random_state.uniform()
            if u > prob:
                # reject ... make state same as previous
                for key in self.state:
                    if type(self.state[key]) is not int:
                        self.state[key][batch_index] = \
                            self.state[key][batch_index-1]
                if self._is_rbsl():
                    # update with prev loglik
                    previous_posterior = self.state['logposterior'][batch_index-1]
                    previous_loglik = previous_posterior - self.state['logprior'][batch_index-1]

                    self.model[self.bsl_name].update_prev_iter_logliks(previous_loglik)

            else:
                # accept
                if batch_index > self.burn_in:
                    self.num_accepted += 1
                if self._is_rbsl():
                    current_posterior = self.state['logposterior'][batch_index]
                    current_loglik = current_posterior - self.state['logprior'][batch_index]
                    self.model[self.bsl_name].update_prev_iter_logliks(current_loglik)
        params = [self.state[p][batch_index] for p in param_names]
        self._report_batch(batch_index, params, batch[self.bsl_name])  # , batch[self.target_name])

        if self._is_rbsl() and self.start_new_chain:
            self.model[self.bsl_name].update_prev_iter_logliks(
                batch[self.bsl_name])

        # delete summaries in state that are not needed for the output
        if batch_index > 0:
            summary_delete = [name for name in self.summary_names if name
                              not in self.output_summaries]
            for s in summary_delete:
                self.state[s][batch_index-1] = None

    def _get_mh_ratio(self, batch_index):
        """Calculate the Metropolis-Hastings ratio.

        Also transforms the parameter range with logit transform if
        needed.

        Parameters
        ----------
        batch_index: int

        """
        current = self.state['logposterior'][batch_index]
        if self._is_rbsl():
            previous_loglik = self.model[self.bsl_name].\
                state['slice_sampler_logliks'][-1]
            if previous_loglik is None:
                previous_loglik = self.state['logposterior'][batch_index-1] - \
                    self.state['logprior'][batch_index-1]
            previous = previous_loglik + self.state['logprior'][batch_index-1]
        else:
            previous = self.state['logposterior'][batch_index-1]

        logp2 = 0
        logit_transform_bound = self.logit_transform_bound
        if logit_transform_bound is not None:
            curr_sample = [self.state[p][batch_index] for p in
                           self.parameter_names]
            prev_sample = [self.state[p][batch_index-1] for p in
                           self.parameter_names]
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

    def prepare_new_batch(self, batch_index):
        """Prepare parameter values for a new batch."""
        self.start_new_chain = batch_index == 0  # old language

        # allow option to use custom param names
        param_names = self.param_names if self.param_names \
            is not None else self.parameter_names

        if self.start_new_chain:
            if self.params0 is not None:
                if isinstance(self.params0, dict):
                    state = self.params0
                else:
                    state = dict(zip(param_names, self.params0))
            else:
                state = self.model.generate(1, param_names,
                                            seed=self.seed)

        for p in param_names:
            if self.start_new_chain:
                self.prior_state[p] = state[p]
            else:
                self.prior_state[p] = self.state[p][batch_index-1]
        state = np.asarray([self.prior_state[p] for p in param_names]).flatten()
        if self.start_new_chain:
            self.prop_state = state.reshape(1, -1)
        else:
            not_in_support = True
            if self.sigma_proposals is None:
                raise ValueError("The random walk proposal covariance must be "
                                 "provided for Metropolis-Hastings sampling.")
            cov = self.sigma_proposals

            while not_in_support:
                self._propagate_state(mean=state, cov=cov,
                                      random_state=self.random_state)
                if np.isfinite(self._prior.logpdf(self.prop_state)):
                    not_in_support = False
                else:
                    # NOTE: increasing cov is a poor solution, if propagate
                    # state is giving infinite prior pdf should consider using
                    # the logit_transform_bound parameter in the BSL class
                    logger.warning('Initial value of chain does not have'
                                   'support. (state: {} cov: {})'.format(
                                        state, cov))
                    cov = cov * 1.01

        params = np.repeat(self.prop_state, axis=0, repeats=self.batch_size)
        batch = arr2d_to_batch(params, param_names)

        # Misspecified BSL needs some params...
        if self._is_rbsl():
            if batch_index > 0:
                ssx_prev = np.column_stack([self.state[p][batch_index-1] for p in
                                            self.summary_names])
                std = np.std(ssx_prev, axis=0)
                sample_mean = np.mean(ssx_prev, axis=0)
                sample_cov = np.cov(ssx_prev, rowvar=False)
                self.model[self.bsl_name].\
                    update_rbsl_operation(std, sample_mean,
                                          sample_cov)

        return batch

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

    def _propagate_state(self, mean, cov=0.01, random_state=None):
        """Logic for random walk proposal. Sets the proposed parameters.

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
            mean_tilde = self._para_logit_transform(mean,
                                                    self.logit_transform_bound)
            sample = scipy_randomGen.rvs(mean=mean_tilde, cov=cov)
            self.prop_state =  \
                self._para_logit_back_transform(sample,
                                                self.logit_transform_bound)
        else:
            self.prop_state = scipy_randomGen.rvs(mean=mean, cov=cov)
        self.prop_state = np.atleast_2d(self.prop_state)

    def select_penalty_helper(self, theta):
        """Get log-likelihoods used in the select penalty module.

        Parameters
        ----------
        theta : np.array
            Theta point to find a log-likelihood estimate

        Returns
        -------
        Log-likelihood vector

        """
        # do minimum initialisation to get 1 iteration
        self.params0 = theta

        if not hasattr(self, 'n_samples'):
            self.n_samples = 1

        self.state['logposterior'] = np.empty(self.n_samples)
        self.state['logprior'] = np.empty(self.n_samples)

        for parameter_name in self.parameter_names:
            self.state[parameter_name] = np.empty(self.n_samples)

        self.set_objective()

        self.iterate()

        # return log-likelihood
        return self.state['logposterior'][0] - self.state['logprior'][0]

    def get_ssx(self, theta):
        """Get simulated summary statistics at theta.

        Parameters
        ----------
        theta : np.array
            Theta point to find a log-likelihood estimate

        Returns
        -------
        ssx : np.array
            Simulated summaries

        """
        # do minimum initialisation to get 1 iteration
        try:
            method = self.model[self.bsl_name].state['original_discrepancy_str']
        except KeyError:
            raise Exception('BSL method not found. Create a new SyntheticLikelihood node')
        self.params0 = theta
        if not hasattr(self, 'n_samples'):
            self.n_samples = 1

        self.state['logposterior'] = np.empty(self.n_samples)
        self.state['logprior'] = np.empty(self.n_samples)

        for parameter_name in self.parameter_names:
            self.state[parameter_name] = np.empty(self.n_samples)

        self.set_objective()

        self.iterate()
        ssx = np.column_stack(tuple([self.state[s][0] for s in
                                     self.summary_names]))

        if method == "semibsl":
            semibsl_fn = \
                self.model[self.bsl_name]['attr_dict']['_operation']
            ssx = semibsl_fn(ssx, whitening="whitening",
                             observed=self.observed)
        return ssx

    def plot_summary_statistics(self, batch_size, theta_point):
        """Plot summary statistics to check for normality.

        Parameters
        ----------
        batch_size : int
            Here refers to number of simulations at theta_point
        theta_point : np.array
            Theta estimate where all simulations are run.

        """
        m = self.model.copy()
        bsl_temp = elfi.BSL(m[self.bsl_name],
                            output_names=self.summary_names,
                            batch_size=batch_size)

        ssx_dict = {}
        bsl_temp.sample(1)
        for output_name in bsl_temp.output_names:
            if output_name in self.summary_names:
                if bsl_temp.state[output_name][0].ndim > 1:
                    n, ns = bsl_temp.state[output_name][0].shape[0:2]
                    for i in range(ns):
                        new_output_name = output_name + '_' + str(i)
                        ssx_dict[new_output_name] =  \
                            bsl_temp.state[output_name][0][:, i]
                else:
                    ssx_dict[output_name] = bsl_temp.state[output_name]

        return vis.plot_summaries(ssx_dict, self.summary_names)

    def plot_covariance_matrix(self, theta, batch_size, corr=False,
                               precision=False, colorbar=True):
        """Plot correlation matrix of summary statistics.

        Check sparsity of covariance (or correlation) matrix.
        Useful to determine if shrinkage estimation could be applied
        which can reduce the number of model simulations required.

        Parameters
        ----------
        theta : np.array
            Theta estimate where all simulations are run.
        batch_size : int
            Number of simulations at theta
        corr : bool
            True -> correlation, False -> covariance
        precision: bool
            True -> precision matrix, False -> covariance/corr.
        precision

        """
        model = self.model.copy()
        if isinstance(theta, dict):
            param_values = theta
        else:
            param_values = dict(zip(model.parameter_names, theta))

        ssx = model.generate(batch_size,
                             outputs=self.summary_names,
                             with_values=param_values)
        ssx = np.vstack([value for value in ssx.values()])
        ssx = ssx.reshape((ssx.shape[0:2]))
        sample_cov = np.cov(ssx, rowvar=False)
        if corr:
            sample_cov = np.corrcoef(sample_cov)  # correlation matrix
        if precision:
            sample_cov = np.linalg.inv(sample_cov)

        fig = plt.figure()
        ax = plt.subplot(111)
        plt.style.use('ggplot')

        cax = ax.matshow(sample_cov)
        if colorbar:
            fig.colorbar(cax)

    def log_SL_stdev(self, theta, batch_size, M):
        """Estimate the standard deviation of the log SL.

        Parameters
        ----------
        theta : np.array
            Theta estimate where all simulations are run.
        batch_size : int
            Number of simulations at theta_point
        M : int
            Number of log-likelihoods to estimate standard deviation
        Returns
        -------
        Estimated standard deviations of log-likelihood

        """
        m = self.model.copy()
        logliks = np.zeros(M)
        for i in range(M):
            bsl_temp = elfi.BSL(m[self.bsl_name],
                                batch_size=batch_size,
                                seed=i)
            logliks[i] = bsl_temp.select_penalty_helper(theta)
        return np.std(logliks)

    def _is_rbsl(self):
        """Ad hoc way of telling if SL target node is for R-BSL."""
        method_str = ""
        if 'original_discrepancy_str' in self.model[self.bsl_name].state:
            method_str = self.model[self.bsl_name].\
                state['original_discrepancy_str']
        return (method_str == "rbsl" or method_str == "misspecbsl")
