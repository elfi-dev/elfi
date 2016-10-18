import scipy as sp

from .utils import stochastic_optimization

class BolfiPosterior():

    def __init__(self, model, threshold, priors=None):
        self.threshold = threshold
        self.model = model
        self.priors = [None] * model.n_var
        self.ML, ML_val = stochastic_optimization(self._neg_unnormalized_loglikelihood_density, self.model.bounds, 10000)
        print("ML parameters: %s" % (self.ML))
        self.MAP, MAP_val = stochastic_optimization(self._neg_unnormalized_logposterior_density, self.model.bounds, 10000)
        print("MAP parameters: %s" % (self.MAP))

    def _unnormalized_loglikelihood_density(self, x):
        mean, var, std = self.model.evaluate(x)
        return sp.stats.norm.logcdf(self.threshold, mean, std)

    def _unnormalized_likelihood_density(self, x):
        return np.exp(self._unnormalized_loglikelihood_density(x))

    def _neg_unnormalized_loglikelihood_density(self, x):
        return -1 * self._unnormalized_loglikelihood_density(x)

    def _unnormalized_logposterior_density(self, x):
        return self._unnormalized_loglikelihood_density(x) + self._logprior_density(x)

    def _unnormalized_posterior_density(self, x):
        return np.exp(self._unnormalized_logposterior_density(x))

    def _neg_unnormalized_logposterior_density(self, x):
        return -1 * self._unnormalized_logposterior_density(x)

    def _logprior_density(self, x):
        logprior_density = 0.0
        for xv, prior in zip(x, self.priors):
            if prior is not None:
                logprior_density += prior.getLogProbDensity(xv)
        return logprior_density

    def _prior_density(self, x):
        return np.exp(self._logprior_density(x))

    def _neg_logprior_density(self, x):
        return -1 * self._logprior_density(x)

    def sample(self):
        return tuple([[v] for v in self.MAP])
