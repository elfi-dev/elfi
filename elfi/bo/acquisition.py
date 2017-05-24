import sys
import logging
import json
import numpy as np

from scipy.stats import truncnorm, uniform, multivariate_normal

from elfi.bo.utils import approx_second_partial_derivative, sum_of_rbf_kernels, \
    stochastic_optimization

logger = logging.getLogger(__name__)

# TODO: make a faster optimization method utilizing parallelization (see e.g. GPyOpt)
# TODO: make use of random_state


class AcquisitionBase:
    """All acquisition functions are assumed to fulfill this interface.
    
    Gaussian noise ~N(0, noise_cov) is added to the acquired points. By default, noise_cov=0.

    Parameters
    ----------
    model : an object with attributes
                input_dim : int
                bounds : tuple of length 'input_dim' of tuples (min, max)
            and methods
                evaluate(x) : function that returns model (mean, var, std)
    n_samples : None or int
        Total number of samples to be sampled, used when part of an
        AcquisitionSchedule object (None indicates no upper bound)
    """
    def __init__(self, model, max_opt_iter=1000, noise_cov=0.):
        self.model = model
        self.max_opt_iter = int(max_opt_iter)

        if isinstance(noise_cov, (float, int)):
            noise_cov = np.eye(self.model.input_dim) * noise_cov
        self.noise_cov = noise_cov

    def evaluate(self, x, t=None):
        """Evaluates the acquisition function value at 'x'

        Returns
        -------
        x : numpy.array
        t : int
            current iteration (starting from 0)
        """
        return NotImplementedError

    def acquire(self, n_values, pending_locations=None, t=None):
        """Returns the next batch of acquisition points.
        
        Gaussian noise ~N(0, self.noise_cov) is added to the acquired points.

        Parameters
        ----------
        n_values : int
            Number of values to return.
        pending_locations : None or numpy 2d array
            If given, acquisition functions may
            use the locations in choosing the next sampling
            location. Locations should be in rows.
        t : int
            Current iteration (starting from 0)

        Returns
        -------
        locations : 2D np.ndarray of shape (n_values, ...)
        """

        logger.debug('Acquiring {} values'.format(n_values))

        obj = lambda x: self.evaluate(x, t)
        minloc, val = stochastic_optimization(obj, self.model.bounds, self.max_opt_iter)

        x = np.tile(minloc, (n_values, 1))

        x += multivariate_normal.rvs(cov=self.noise_cov, size=n_values).reshape((n_values, -1))

        # make sure the acquired points stay within bounds
        for ii in range(self.model.input_dim):
            x[:, ii] = np.clip(x[:, ii], *self.model.bounds[ii])

        return x


class LCBSC(AcquisitionBase):
    """Lower Confidence Bound Selection Criterion. Srinivas et al. call it GP-LCB.
    
    Parameter delta must be in (0, 1). The theoretical upper bound for total regret in 
    Srinivas et al. has a probability greater or equal to 1 - delta, so values of delta 
    very close to 1 do not make much sense in that respect.
    
    Delta is roughly the exploitation tendency of the acquisition function.

    References
    ----------
    N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger. Gaussian
    process optimization in the bandit setting: No regret and experimental design. In
    Proc. International Conference on Machine Learning (ICML), 2010
    
    E. Brochu, V.M. Cora, and N. de Freitas. A tutorial on Bayesian optimization of expensive
    cost functions, with application to active user modeling and hierarchical reinforcement
    learning. arXiv:1012.2599, 2010.
    
    Notes
    -----
    The formula presented in Brochu (pp. 15) seems to be from Srinivas et al. Theorem 2.
    However, instead of having t**(d/2 + 2) in \beta_t, it seems that the correct form 
    would be t**(2d + 2).
    """
    
    def __init__(self, *args, delta=.1, **kwargs):
        super(LCBSC, self).__init__(*args, **kwargs)
        if delta <= 0 or delta >= 1:
            raise ValueError('Parameter delta must be in the interval (0,1)')
        self.delta = delta

    def beta(self, t):
        # Start from 0
        t += 1
        d = self.model.input_dim
        return 2*np.log(t**(2*d + 2) * np.pi**2 / (3*self.delta))

    def evaluate(self, x, t=None):
        """ Lower confidence bound selection criterion = mean - sqrt(\beta_t) * std """
        if not isinstance(t, int):
            raise ValueError("Parameter 't' should be an integer.")

        mean, var = self.model.predict(x, noiseless=True)
        return mean - np.sqrt(self.beta(t) * var)


class UniformAcquisition(AcquisitionBase):

    def acquire(self, n_values, pending_locations=None, t=None):
        bounds = np.stack(self.model.bounds)
        return uniform(bounds[:,0], bounds[:,1] - bounds[:,0])\
            .rvs(size=(n_values, self.model.input_dim))
