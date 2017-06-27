import logging

import numpy as np
from scipy.stats import uniform, truncnorm

from elfi.methods.bo.utils import minimize


logger = logging.getLogger(__name__)


class AcquisitionBase:
    """All acquisition functions are assumed to fulfill this interface.
    
    Gaussian noise ~N(0, noise_var) is added to the acquired points. By default,
    noise_var=0. You can define a different variance for the separate dimensions.

    Parameters
    ----------
    model : an object with attributes
                input_dim : int
                bounds : tuple of length 'input_dim' of tuples (min, max)
            and methods
                evaluate(x) : function that returns model (mean, var, std)
    prior
        By default uniform distribution within model bounds.
    n_inits : int, optional
        Number of initialization points in internal optimization.
    max_opt_iters : int, optional
        Max iterations to optimize when finding the next point.
    noise_var : float or np.array, optional
        Acquisition noise variance for adding noise to the points near the optimized
        location. If array, must be 1d specifying the variance for different dimensions.
        Default: no added noise.
    seed : int
    """
    def __init__(self, model, prior=None, n_inits=10, max_opt_iters=1000, noise_var=None,
                 seed=None):

        self.model = model
        self.prior = prior
        self.n_inits = int(n_inits)
        self.max_opt_iters = int(max_opt_iters)

        if noise_var is not None and np.asanyarray(noise_var).ndim > 1:
            raise ValueError("Noise variance must be a float or 1d vector of variances "
                             "for the different input dimensions.")
        self.noise_var = noise_var

        self.random_state = np.random if seed is None else np.random.RandomState(seed)

    def evaluate(self, x, t=None):
        """Evaluates the acquisition function value at 'x'.

        Parameters
        ----------
        x : numpy.array
        t : int
            current iteration (starting from 0)
        """
        raise NotImplementedError

    def evaluate_gradient(self, x, t=None):
        """Evaluates the gradient of acquisition function value at 'x'.

        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        raise NotImplementedError

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
            Current iteration (starting from 0).

        Returns
        -------
        locations : 2D np.ndarray of shape (n_values, ...)
        """
        logger.debug('Acquiring {} values'.format(n_values))

        obj = lambda x: self.evaluate(x, t)
        grad_obj = lambda x: self.evaluate_gradient(x, t)
        xhat, _ = minimize(obj, self.model.bounds, grad_obj, self.prior, self.n_inits,
                           self.max_opt_iters, random_state=self.random_state)
        x = np.tile(xhat, (n_values, 1))

        # Add some noise for more efficient fitting of GP
        if self.noise_var is not None:
            noise_var = np.asanyarray(self.noise_var)
            if noise_var.ndim == 0:
                noise_var = np.tile(noise_var, self.model.input_dim)

            for i in range(self.model.input_dim):
                std = np.sqrt(noise_var[i])
                if std == 0:
                    continue
                xi = x[:, i]
                a = (self.model.bounds[i][0] - xi) / std
                b = (self.model.bounds[i][1] - xi) / std
                x[:, i] = truncnorm.rvs(a, b, loc=xi, scale=std, size=n_values,
                                        random_state=self.random_state)

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
    
    def __init__(self, *args, delta=0.1, **kwargs):
        super(LCBSC, self).__init__(*args, **kwargs)
        if delta <= 0 or delta >= 1:
            raise ValueError('Parameter delta must be in the interval (0,1)')
        self.delta = delta

    def _beta(self, t):
        # Start from 0
        t += 1
        d = self.model.input_dim
        return 2*np.log(t**(2*d + 2) * np.pi**2 / (3*self.delta))

    def evaluate(self, x, t):
        """Lower confidence bound selection criterion: 
        
        mean - sqrt(\beta_t) * std
        
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        mean, var = self.model.predict(x, noiseless=True)
        return mean - np.sqrt(self._beta(t) * var)

    def evaluate_gradient(self, x, t):
        """Gradient of the lower confidence bound selection criterion.
        
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        mean, var = self.model.predict(x, noiseless=True)
        grad_mean, grad_var = self.model.predictive_gradients(x)

        return grad_mean - 0.5 * grad_var * np.sqrt(self._beta(t) / var)


class UniformAcquisition(AcquisitionBase):

    def acquire(self, n_values, pending_locations=None, t=None):
        bounds = np.stack(self.model.bounds)
        return uniform(bounds[:,0], bounds[:,1] - bounds[:,0])\
            .rvs(size=(n_values, self.model.input_dim), random_state=self.random_state)
