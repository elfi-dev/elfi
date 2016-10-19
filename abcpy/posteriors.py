import scipy as sp

from .utils import stochastic_optimization

class Posterior():
    """Container for the posterior that an .inter() method returns.

    Attributes
    ----------
    samples : list-type
        Pre-computed samples from the posterior.
    """

    def __init__(self):
        self.samples = list()

    def __getitem__(self, idx):
        """ Returns samples from posterior.

        Parameters
        ----------
        idx : slice-type
            Indexes of samples to return

        Returns
        -------
        list-type
            samples
        """
        return self.samples[idx]

    def pdf(self, x, norm=False):
        """ Returns probability density at x.

        Parameters
        ----------
        x : numpy 1d array
            Location in parameter space.
        norm : bool
            True: density value needs to be normalized.
            False: density value may be unnormalized.

        Returns
        -------
        float or None
            probability density value, or None if not implemented
        """
        return None

    def logpdf(self, x, norm=False):
        """ Returns log probability density at x.

        Parameters
        ----------
        x : numpy 1d array
            Location in parameter space.
        norm : bool
            True: density value needs to be normalized.
            False: density value may be unnormalized.

        Returns
        -------
        float or None
            log probability density value, or None if not implemented
        """
        return None


class BolfiPosterior(Posterior):

    def __init__(self, model, threshold, priors=None):
        super(BolfiPosterior, self).__init__()
        self.threshold = threshold
        self.model = model
        self.priors = [None] * model.n_var
        self.ML, ML_val = stochastic_optimization(self._neg_unnormalized_loglikelihood_density, self.model.bounds, 10000)
        print("ML parameters: %s" % (self.ML))
        self.MAP, MAP_val = stochastic_optimization(self._neg_unnormalized_logposterior_density, self.model.bounds, 10000)
        print("MAP parameters: %s" % (self.MAP))

    def logpdf(self, x, norm=False):
        if norm is True:
            return None
        return self._unnormalized_loglikelihood_density(x) + self._logprior_density(x)

    def pdf(self, x, norm=False):
        if norm is True:
            return None
        return np.exp(self._unnormalized_logposterior_density(x))

    def __getitem__(self, idx):
        return tuple([[v]*len(idx) for v in self.MAP])

    def _unnormalized_loglikelihood_density(self, x):
        mean, var, std = self.model.evaluate(x)
        return sp.stats.norm.logcdf(self.threshold, mean, std)

    def _unnormalized_likelihood_density(self, x):
        return np.exp(self._unnormalized_loglikelihood_density(x))

    def _neg_unnormalized_loglikelihood_density(self, x):
        return -1 * self._unnormalized_loglikelihood_density(x)

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

