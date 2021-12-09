"""This module contains BSL classes"""

__all__ = ['BSL']

import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np

import elfi.client
import elfi.visualization.visualization as vis
from elfi.methods.results import BslSample
from elfi.model.extensions import ModelPrior
from elfi.methods.utils import arr2d_to_batch
from elfi.methods.inference.samplers import Sampler


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

    def __init__(self, model, discrepancy_name=None,
                 observed=None, output_names=None,
                 parameter_names=None,
                 # chains=1, chain_length=5000,
                 #  tkde=None,
                 batch_size=1, seed=None,
                 **kwargs):
        """Initialize the BSL sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        summary_names: array-like, str
            Names of the summary nodes in the model that are to be used
            for the BSL parametric approximation
        observed : np.array, optional
            If not given defaults to observed generated in model.
        output_names :
            Additional outputs from the model to be included in the inference
            result, e.g. corresponding summaries to the acquired samples
        method : str, optional
            Specifies the bsl method to approximate the likelihood.
            Defaults to "bsl".
        shrinkage : str, optional
            The shrinkage method to be used with the penalty param.
        penalty : float, optional
            The penalty value to used for the specified shrinkage method.
            Must be between zero and one when using shrinkage method "Warton".
        whitening : np.array of shape (m x m) - m = num of summary statistics
            The whitening matrix that can be used to estimate the sample
            covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
            transformation helps decorrelate the summary statistics allowing
            for heaving shrinkage to be applied (hence smaller batch_size).
        type_misspec : str, optional
            Needed when using the "misspecBsl" method. Options are either mean
            or variance.
        logitTransformBound : np.array of lists, optional
            Specifies the upper and lower bounds of parameters if a logit
            transformation is ued on the parameter space. First column is lower
            bound and second upper bound. Infinite bounds are supported.
        tkde : str, optional  -- # TODO: functionality in progress
            Sets the transformation depending on the data shape.
            tkde0 - log, tkde1, tkde2, tkde3...
        standardise: bool, optional
            Used with "glasso" shrinkage. Defaults to False.
        batch_size : int, optional
            The number of parameter evaluations in each pass through the
            ELFI graph. When using a vectorized simulator, using a suitably
            large batch_size can provide a significant performance boost.
            In the context of BSL, this is the number of simulations for 1
            parametric approximation of the likelihood.
        seed : int, optional
            Seed for the data generation from the ElfiModel
        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        if output_names is None:
            output_names = []
        summary_names = [summary.name for summary in
                         model[discrepancy_name].parents]
        self.summary_names = summary_names

        output_summaries = [name for name in summary_names if name in
                            output_names]

        output_names = output_names + summary_names + model.parameter_names + \
            [discrepancy_name]
        self.discrepancy_name = discrepancy_name
        super(BSL, self).__init__(
            model, output_names, batch_size, seed, **kwargs)
        # self.summary_names = summary_names
        if isinstance(summary_names, str):
            self.summary_names = np.array([summary_names])
        self.observed = observed
        self.prior_state = dict()
        # self.chain_length = chain_length
        # self.chains = chains
        self.prop_state = dict()
        self._prior = ModelPrior(model)
        self.num_accepted = 0
        # self.tkde = tkde
        self.output_summaries = output_summaries

        for name in self.output_names:
            if name not in self.parameter_names:
                self.state[name] = list()

        if self.observed is None:
            self.observed = self._get_observed_summary_values()

    def _get_observed_summary_values(self):
        """Gets the observed values for summary statistics

        Returns:
        obs_ss : np.ndarray
            The observed summary statistic vector.
        """
        obs_ss = [self.model[summary_name].observed for summary_name in
                  self.summary_names]
        obs_ss = np.column_stack(obs_ss)
        return obs_ss

    def sample(self, n_samples, burn_in=0, params0=None, sigma_proposals=None,
               logitTransformBound=None,
               *args, **kwargs):
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
        Returns
        -------
        BslSample
        """
        self.params0 = params0
        self.sigma_proposals = sigma_proposals
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.logitTransformBound = logitTransformBound
        self.state['logposterior'] = np.empty(self.n_samples)
        self.state['logprior'] = np.empty(self.n_samples)

        for parameter_name in self.parameter_names:
            self.state[parameter_name] = np.empty(self.n_samples)

        return super().sample(n_samples)

    def set_objective(self, *args, **kwargs):
        """Set objective for inference.
        """
        self.objective['batch_size'] = self.batch_size
        if hasattr(self, 'n_samples'):
            self.objective['n_batches'] = self.n_samples
            self.objective['n_sim'] = self.n_samples * self.batch_size

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample
        """
        outputs = dict()
        samples_all = dict()
        index_array = np.array(range(self.burn_in, self.n_samples))
        burn_in_mask = index_array

        # for i in range(self.chains - 1):  # old logic for chains
        # burn_in_mask = np.append(burn_in_mask, index_array + self.n_samples * (i + 1))

        binary_array = np.zeros(burn_in_mask[-1] + 1)
        binary_array[burn_in_mask] = 1

        summary_delete = [name for name in self.summary_names if name not in self.output_summaries]

        for p in self.output_names:
            if p in summary_delete:
                continue
            sample = [state_p for ii, state_p in enumerate(self.state[p])]
            samples_all[p] = np.array(sample)
            output = [state_p for ii, state_p in enumerate(self.state[p])
                      if ii >= self.burn_in]
            outputs[p] = np.array(output)

        acc_rate = self.num_accepted/(self.n_samples - self.burn_in)

        return BslSample(
             samples_all=samples_all,  # includes burn_in in samples
             outputs=outputs,
             acc_rate=acc_rate,
             burn_in=self.burn_in,
             **self._extract_result_kwargs()
        )

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch: dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(BSL, self).update(batch, batch_index)
        random_state = np.random.RandomState(self.seed+batch_index)
        ssx = np.column_stack([batch[name] for name in self.summary_names])

        dim1, dim2 = ssx.shape[0:2]
        ssx = ssx.reshape((dim1, dim2))

        prior_vals = [batch[p][0] for p in self.parameter_names]
        prior_vals = np.array(prior_vals)
        self.state['logprior'][batch_index] = self._prior.logpdf(prior_vals)
        # self._evaluate_logpdf(batch_index, prior_vals, ssx)
        self.state['logposterior'][batch_index] = \
            batch[self.discrepancy_name] + self.state['logprior'][batch_index]
        for s in self.output_names:
            if s not in self.parameter_names:
                self.state[s].append(batch[s])

        tmp = {p: self.prop_state[0, i] for i, p in
               enumerate(self.parameter_names)}

        for p in self.parameter_names:
            self.state[p][batch_index] = tmp[p]

        if not self.start_new_chain:
            l_ratio = self._get_mh_ratio(batch_index)
            prob = np.minimum(1.0, l_ratio)
            u = random_state.uniform()
            if u > prob:
                # reject ... make state same as previous
                for key in self.state:
                    if type(self.state[key]) is not int:
                        self.state[key][batch_index] = \
                            self.state[key][batch_index-1]
            else:
                # accept
                if batch_index > self.burn_in:
                    self.num_accepted += 1
                    # print('self.acc_rate', self.num_accepted/(batch_index -
                    # self.burn_in))

        # delete summaries in state that are not needed for the output
        if batch_index > 0:
            summary_delete = [name for name in self.summary_names if name
                              not in self.output_summaries]
            for s in summary_delete:
                self.state[s][batch_index-1] = None

    def _get_mh_ratio(self, batch_index):
        """Calculate the Metropolis-Hastings ratio and transform the parameter
           range with logit transform if needed.

           Parameters
           ----------
           batch_index: int

        """
        current = self.state['logposterior'][batch_index]
        previous = self.state['logposterior'][batch_index-1]
        logp2 = 0
        logitTransformBound = self.logitTransformBound
        if logitTransformBound is not None:
            curr_sample = [self.state[p][batch_index] for p in
                           self.parameter_names]
            prev_sample = [self.state[p][batch_index-1] for p in
                           self.parameter_names]
            curr_sample = np.array(curr_sample)
            prev_sample = np.array(prev_sample)
            logp2 = self._jacobian_logit_transform(curr_sample,
                                                   logitTransformBound) - \
                self._jacobian_logit_transform(prev_sample,
                                               logitTransformBound)
        return np.exp(logp2 + current - previous)

    def prepare_new_batch(self, batch_index):
        """Prepare parameter values for a new batch."""

        # self.start_new_chain = (batch_index % self.chain_length) == 0
        self.start_new_chain = batch_index == 0  # old language

        if self.start_new_chain:
            if self.params0 is not None:
                state = dict(zip(self.parameter_names, self.params0))
            else:
                state = self.model.generate(1, self.parameter_names)

        for p in self.parameter_names:
            if self.start_new_chain:
                self.prior_state[p] = state[p]
            else:
                self.prior_state[p] = self.state[p][batch_index-1]
        state = np.asarray([self.prior_state[p] for p in self.parameter_names]).flatten()
        if self.start_new_chain:
            self.prop_state = state.reshape(1, -1)
        else:
            not_in_support = True
            if self.sigma_proposals is None:
                raise ValueError("Gaussian proposal standard deviations have"
                                 "to be provided for Metropolis-sampling.")

            cov = self.sigma_proposals
            random_state = np.random.RandomState(self.seed+batch_index)
            while not_in_support:
                self._propagate_state(mean=state, cov=cov,
                                      random_state=random_state)
                if np.isfinite(self._prior.logpdf(self.prop_state)):
                    not_in_support = False
                else:
                    # increasing cov is a poor solution, if propagate state
                    # is giving infinite prior pdf should consider using
                    # the logitTransformBound parameter in the BSL class
                    cov = cov * 1.01

        params = np.repeat(self.prop_state, axis=0, repeats=self.batch_size)
        batch = arr2d_to_batch(params, self.parameter_names)

        # Misspecified BSL needs some params...
        if 'logliks' in self.model[self.discrepancy_name].state:
            # TODO! CHECK SAVES ONLY MISSPEC AND DEL AS GO
            if batch_index > 0:
                loglik = self.state['logposterior'][batch_index-1] - \
                            self.state['logprior'][batch_index-1]
                ssx_prev = np.column_stack([self.state[p][batch_index-1] for p in
                                            self.summary_names])
                std = np.std(ssx_prev, axis=0)
                sample_mean = np.mean(ssx_prev, axis=0)
                sample_cov = np.cov(ssx_prev, rowvar=False)
                self.model[self.discrepancy_name].\
                    update_misspecbsl_operation(loglik, std, sample_mean,
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
        """Find jacobian of logit transform
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
        if self.logitTransformBound is not None:
            mean_tilde = self._para_logit_transform(mean,
                                                    self.logitTransformBound)
            sample = scipy_randomGen.rvs(mean=mean_tilde, cov=cov)
            self.prop_state =  \
                self._para_logit_back_transform(sample,
                                                self.logitTransformBound)
        else:
            self.prop_state = scipy_randomGen.rvs(mean=mean, cov=cov)
        self.prop_state = np.atleast_2d(self.prop_state)

    def select_penalty_helper(self, theta):
        """Helper to get log-likelihoods used in the select penalty module.

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
        """
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
        method = self.model[self.discrepancy_name].state['original_discrepancy_str']
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
                self.model[self.discrepancy_name]['attr_dict']['_operation']
            ssx = semibsl_fn(ssx, whitening="whitening",
                             observed=self.observed)
        return ssx

    def plot_summary_statistics(self, batch_size, theta_point):
        """Helper function to plot summary statistics of model.
           Useful to check for normality of summary statistics.

        Parameters
        ----------
        batch_size : int
            Here refers to number of simulations at theta_point
        theta_point : np.array
            Theta estimate where all simulations are run.
        """
        m = self.model.copy()
        bsl_temp = elfi.BSL(m[self.discrepancy_name],
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

    def plot_correlation_matrix(self, theta, batch_size, corr=False,
                                precision=False):
        """Check sparsity of covariance (or correlation) matrix.
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
        ssx = self.get_ssx(theta)
        ssx = ssx.reshape((ssx.shape[0:2]))
        sample_cov = np.cov(ssx, rowvar=False)
        if corr:
            sample_cov = np.corrcoef(sample_cov)  # correlation matrix
        if precision:
            sample_cov = np.linalg.inv(sample_cov)
        plt.matshow(sample_cov)

    def log_SL_stdev(self, theta, batch_size, M):
        """
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
        Standard deviations of log-likelihood
        """
        m = self.model.copy()
        logliks = np.zeros(M)
        for i in range(M):
            bsl_temp = elfi.BSL(m[self.discrepancy_name],
                                batch_size=batch_size,
                                seed=i)  # TODO? make more broad
            logliks[i] = bsl_temp.select_penalty_helper(theta) 
        return np.std(logliks)
