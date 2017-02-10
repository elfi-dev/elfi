import logging
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt

from .utils import stochastic_optimization

logger = logging.getLogger(__name__)

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
        """Returns samples from posterior.

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
        """Returns probability density at x.

        Parameters
        ----------
        x : numpy 1d array
            Location in parameter space.
        norm : bool
            True: density value needs to be normalized.
            False: density value may be unnormalized.

        Returns
        -------
        float
            probability density value
        """
        raise NotImplementedError("Normalized posterior not implemented")

    def logpdf(self, x, norm=False):
        """Returns log probability density at x.

        Parameters
        ----------
        x : numpy 1d array
            Location in parameter space.
        norm : bool
            True: density value needs to be normalized.
            False: density value may be unnormalized.

        Returns
        -------
        float
            log probability density value
        """
        raise NotImplementedError("Normalized logposterior not implemented")

    def plot(self, *args, **kwargs):
        """Simple matplotlib printout of the posterior for convenience.

        Parameters
        ----------
        model specific
        """
        pass


class BolfiPosterior(Posterior):

    def __init__(self, model, threshold, priors=None, max_opt_iters=10000):
        super(BolfiPosterior, self).__init__()
        self.threshold = threshold
        self.model = model
        if self.threshold is None:
            minloc, minval = stochastic_optimization(self.model.eval_mean, self.model.bounds, max_opt_iters)
            self.threshold = minval
            logger.info("Using minimum value of discrepancy estimate mean (%.4f) as threshold" % (self.threshold))
        self.priors = [None] * model.input_dim
        self.ML, self.ML_val = stochastic_optimization(self._neg_unnormalized_loglikelihood_density, self.model.bounds, max_opt_iters)
        self.MAP, self.MAP_val = stochastic_optimization(self._neg_unnormalized_logposterior_density, self.model.bounds, max_opt_iters)

    def logpdf(self, x, norm=False):
        if norm is True:
            raise NotImplementedError("Normalized logposterior not implemented")
        return self._unnormalized_loglikelihood_density(x) + self._logprior_density(x)

    def pdf(self, x, norm=False):
        if norm is True:
            raise NotImplementedError("Normalized posterior not implemented")
        return np.exp(self.logpdf(x))

    def __getitem__(self, idx):
        return tuple([[v]*len(idx) for v in self.MAP])

    def _unnormalized_loglikelihood_density(self, x):
        mean, var, std = self.model.evaluate(x)
        if mean is None or std is None:
            raise ValueError("Unable to evaluate model at %s" % (x))
        return sp.stats.norm.logcdf(self.threshold, mean, std)

    def _unnormalized_likelihood_density(self, x):
        return np.exp(self._unnormalized_loglikelihood_density(x))

    def _neg_unnormalized_loglikelihood_density(self, x):
        return -1 * self._unnormalized_loglikelihood_density(x)

    def _neg_unnormalized_logposterior_density(self, x):
        return -1 * self.logpdf(x)

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

    def plot(self, norm=False):
        if len(self.model.bounds) == 1:
            mn = self.model.bounds[0][0]
            mx = self.model.bounds[0][1]
            dx = (mx - mn) / 200.0
            x = np.arange(mn, mx, dx)
            pd = np.zeros(len(x))
            for i in range(len(x)):
                pd[i] = self.pdf([x[i]], norm)
            plt.figure()
            plt.plot(x, pd)
            plt.xlim(mn, mx)
            plt.ylim(0.0, max(pd)*1.05)
            plt.show()

