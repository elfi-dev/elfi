import numpy as np
import scipy.stats as ss
from copy import copy

from .methods import Rejection
from .distributions import Prior
from .utils import weighted_var


class SMC(Rejection):
    """Likelihood-free sequential Monte Carlo sampler.

    Based on Algorithm 4 in:
    Jean-Michel Marin, Pierre Pudlo, Christian P Robert, and Robin J Ryder:
    Approximate bayesian computational methods, Statistics and Computing,
    22(6):1167–1180, 2012.

    Attributes
    ----------
    (as in Rejection)

    See Also
    --------
    `Rejection` : Basic rejection sampling.
    """

    def infer(self, n_populations, schedule):
        """Run SMC-ABC sampler.

        Parameters
        ----------
        n_populations : int
            Number of particle populations to iterate over.
        schedule : iterable of floats
            Acceptance quantiles for particle populations.

        Returns
        -------
        A dictionary with items:
        samples : list of np.arrays
            Samples from the posterior distribution of each parameter.
        samples_history : list of lists of np.arrays
            Samples from previous populations.
        weighted_sds_history : list of lists of floats
            Weighted standard deviations from previous populations.
        """

        # initialize with rejection sampling
        result = super(SMC, self).infer(quantile=schedule[0])
        parameters = result['samples']
        weights = np.ones(self.n_samples)

        # save original priors
        orig_priors = [p for p in self.parameter_nodes]

        params_history = []
        weighted_sds_history = []
        for tt in range(1, n_populations):
            params_history.append(parameters)

            weights /= np.sum(weights)  # normalize weights here

            # calculate weighted standard deviations
            weighted_sds = [ np.sqrt( 2. * weighted_var(p, weights) )
                             for p in parameters ]
            weighted_sds_history.append(weighted_sds)

            # set new prior distributions based on previous samples
            for ii, p in enumerate(self.parameter_nodes):
                new_prior = Prior(p.name, _SMC_Distribution, parameters[ii],
                                  weighted_sds[ii], weights)
                self.parameter_nodes[ii] = p.change_to(new_prior,
                                                       transfer_parents=False,
                                                       transfer_children=True)

            # rejection sampling with the new priors
            result = super(SMC, self).infer(quantile=schedule[tt])
            parameters = result['samples']

            # calculate new unnormalized weights for parameters
            weights_new = np.ones(self.n_samples)
            for ii in range(self.n_params):
                weights_denom = np.sum(weights *
                                       self.parameter_nodes[ii].pdf(parameters[ii]))
                weights_new *= orig_priors[ii].pdf(parameters[ii]) / weights_denom
                # weights_new *= orig_prior_pdfs[ii](parameters[ii]) / weights_denom
            weights = weights_new

        # revert to original priors
        self.parameter_nodes = [p.change_to(orig_priors[ii],
                                            transfer_parents=False,
                                            transfer_children=True)
                                for ii, p in enumerate(self.parameter_nodes)]

        return {'samples': parameters,
                'samples_history': params_history,
                'weighted_sds_history': weighted_sds_history}


class _SMC_Distribution(ss.rv_continuous):
    """Distribution that samples near previous values of parameters.
    Used in SMC as priors for subsequent particle populations.
    """
    def rvs(current_params, weighted_sd, weights, random_state, size=1):
        selections = random_state.choice(np.arange(current_params.shape[0]), size=size, p=weights)
        params = current_params[selections] + \
                 ss.norm.rvs(scale=weighted_sd, size=size, random_state=random_state)
        return params

    def pdf(params, current_params, weighted_sd, weights):
        return ss.norm.pdf(params, current_params, weighted_sd)
