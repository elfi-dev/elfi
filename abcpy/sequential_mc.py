import numpy as np
import scipy.stats as ss
import numpy.random as npr
from copy import copy

from .methods import Rejection
from .distributions import ScipyPrior


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
        weights = np.ones(self.n_samples)

        # save original prior pdfs
        orig_prior_pdfs = [copy(p.pdf) for p in self.parameter_nodes]

        params_history = []
        for tt in range(1, n_populations):
            params_history.append( list(result['samples']) )

            weights /= np.sum(weights)  # normalize weights here
            weighted_sds = [ np.sqrt( 2. * np.average(
                             (p[:,0] - np.average(p[:,0], weights=weights))**2,
                                                      weights=weights) )
                             for p in self.parameters ]

            # set new prior distributions based on previous samples
            for ii, p in enumerate(self.parameter_nodes):
                # p.replace_by(ScipyPrior(p.name+'_new', ss.norm,
                    # 1, 3))
                p.replace_by(ScipyPrior(p.name, SMC_Distribution,
                    self.parameters[ii][:,0].copy(), weighted_sds[ii], weights))

            # rejection sampling with the new priors
            # threshold = max(np.percentile(self.distances, p_quantile*100),
            #                 schedule[tt])
            result = super(SMC, self).infer(quantile=schedule[tt])

            # calculate new unnormalized weights for parameters
            # TODO: is this correct in multi-dimensional case?
            weights = np.ones(self.n_samples)
            for ii in range(self.n_params):
                weights_denom = np.sum(weights *
                                       self.parameter_nodes[ii].pdf(self.parameters[ii]))
                weights *= orig_prior_pdfs[ii](self.parameters[ii][:,0]) / weights_denom

        return {'samples': self.parameters, 'samples_history': params_history}


class SMC_Distribution(ss.rv_continuous):
    """
    Distribution that samples near previous values.
    """
    def rvs(current_params, weighted_sd, weights, size=1, random_state=None):
        selections = npr.choice(np.arange(current_params.shape[0]), size=size, p=weights)
        params = current_params[selections] + \
                 ss.norm.rvs(scale=weighted_sd, size=size)
        return params

    def pdf(params, current_params, weighted_sd, weights):
        return ss.norm.pdf(params, current_params, weighted_sd)
