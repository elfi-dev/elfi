import numpy as np
import scipy.stats as ss
from copy import copy

from .methods import Rejection
from .distributions import Prior


class SMC(Rejection):
    """
    Likelihood-free sequential Monte Carlo sampler.

    Based on Algorithm 4 in:
    Jean-Michel Marin, Pierre Pudlo, Christian P Robert, and Robin J Ryder:
    Approximate bayesian computational methods, Statistics and Computing,
    22(6):1167–1180, 2012.
    """

    def infer(self, n_populations, schedule):
        """
        Run SMC-ABC sampler.
        """

        # initialize with rejection sampling
        result = super(SMC, self).infer(quantile=schedule[0])
        parameters = result['samples']
        weights = np.ones(self.n_samples)

        # save original prior pdfs
        orig_prior_pdfs = [copy(p.pdf) for p in self.parameter_nodes]

        params_history = []
        for tt in range(1, n_populations):
            params_history.append( list(parameters) )

            weights /= np.sum(weights)  # normalize weights here
            weighted_sds = [ np.sqrt( 2. * np.average(
                             (p - np.average(p, weights=weights))**2,
                                                      weights=weights) )
                             for p in parameters ]

            # set new prior distributions based on previous samples
            self.parameter_nodes = [ p.change_to( Prior(p.name, SMC_Distribution,
                parameters[ii].copy(), weighted_sds[ii], weights ),
                transfer_parents=False, transfer_children=True)
                                    for ii, p in enumerate(self.parameter_nodes) ]

            # rejection sampling with the new priors
            # threshold = max(np.percentile(self.distances, p_quantile*100),
            #                 schedule[tt])
            result = super(SMC, self).infer(quantile=schedule[tt])
            parameters = result['samples']

            # calculate new unnormalized weights for parameters
            # TODO: is this correct in multi-dimensional case?
            weights_old = weights.copy()
            weights = np.ones(self.n_samples)
            for ii in range(self.n_params):
                weights_denom = np.sum(weights_old *
                                       self.parameter_nodes[ii].pdf(parameters[ii]))
                weights *= orig_prior_pdfs[ii](parameters[ii]) / weights_denom

        return {'samples': parameters, 'samples_history': params_history}


class SMC_Distribution(ss.rv_continuous):
    """
    Distribution that samples near previous values.
    """
    def rvs(current_params, weighted_sd, weights, size=1, random_state=None):
        selections = random_state.choice(np.arange(current_params.shape[0]), size=size, p=weights)
        params = current_params[selections] + \
                 ss.norm.rvs(scale=weighted_sd, size=size, random_state=random_state)
        return params

    def pdf(params, current_params, weighted_sd, weights):
        return ss.norm.pdf(params, current_params, weighted_sd)
