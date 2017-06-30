import logging

import numpy as np
from scipy.stats import uniform, truncnorm

from elfi.methods.bo.utils import minimize


logger = logging.getLogger(__name__)


class AcquisitionBase:
    """All acquisition functions are assumed to fulfill this interface.
    
    Gaussian noise ~N(0, self.noise_var) is added to the acquired points. By default,
    noise_var=0. You can define a different variance for the separate dimensions.

    """
    def __init__(self, model, prior=None, n_inits=10, max_opt_iters=1000, noise_var=None,
                 exploration_rate=10, seed=None):
        """

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
        exploration_rate : float, optional
            Exploration rate of the acquisition function (if supported)
        seed : int, optional
            Seed for getting consistent acquisition results. Used in getting random
            starting locations in acquisition function optimization.
        """

        self.model = model
        self.prior = prior
        self.n_inits = int(n_inits)
        self.max_opt_iters = int(max_opt_iters)

        if noise_var is not None and np.asanyarray(noise_var).ndim > 1:
            raise ValueError("Noise variance must be a float or 1d vector of variances "
                             "for the different input dimensions.")
        self.noise_var = noise_var
        self.exploration_rate = exploration_rate
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

    def acquire(self, n, t=None):
        """Returns the next batch of acquisition points.

        Gaussian noise ~N(0, self.noise_var) is added to the acquired points.

        Parameters
        ----------
        n : int
            Number of acquisition points to return.
        t : int
            Current acq_batch_index (starting from 0).
        random_state : np.random.RandomState, optional

        Returns
        -------
        x : np.ndarray
            The shape is (n_values, input_dim)
        """
        logger.debug('Acquiring the next batch of {} values'.format(n))

        # Optimize the current minimum
        obj = lambda x: self.evaluate(x, t)
        grad_obj = lambda x: self.evaluate_gradient(x, t)
        xhat, _ = minimize(obj, self.model.bounds, grad_obj, self.prior, self.n_inits,
                           self.max_opt_iters, random_state=self.random_state)

        # Create n copies of the minimum
        x = np.tile(xhat, (n, 1))
        # Add noise for more efficient fitting of GP
        x = self._add_noise(x)

        return x

    def _add_noise(self, x):
        # Add noise for more efficient fitting of GP
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
                x[:, i] = truncnorm.rvs(a, b, loc=xi, scale=std, size=len(x),
                                        random_state=self.random_state)

        return x


class LCBSC(AcquisitionBase):
    """Lower Confidence Bound Selection Criterion. Srinivas et al. call it GP-LCB.

    LCBSC uses the parameter delta which is here equivalent to 1/exploration_rate.

    Parameter delta should be in (0, 1) for the theoretical results to hold. The
    theoretical upper bound for total regret in Srinivas et al. has a probability greater
    or equal to 1 - delta, so values of delta very close to 1 or over it do not make much
    sense in that respect.

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
    
    def __init__(self, *args, delta=None, **kwargs):
        """

        Parameters
        ----------
        args
        delta : float, optional
            In between (0, 1). Default is 1/exploration_rate. If given, overrides the
            exploration_rate.
        kwargs
        """
        if delta is not None:
            if delta <= 0 or delta >= 1:
                logger.warning('Parameter delta should be in the interval (0,1)')
            kwargs['exploration_rate'] = 1/delta

        super(LCBSC, self).__init__(*args, **kwargs)

    @property
    def delta(self):
        return 1/self.exploration_rate

    def _beta(self, t):
        # Start from 0
        t += 1
        d = self.model.input_dim
        return 2*np.log(t**(2*d + 2) * np.pi**2 / (3*self.delta))

    def evaluate(self, x, t=None):
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

    def evaluate_gradient(self, x, t=None):
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

    def acquire(self, n, t=None):
        bounds = np.stack(self.model.bounds)
        return uniform(bounds[:,0], bounds[:,1] - bounds[:,0])\
            .rvs(size=(n, self.model.input_dim), random_state=self.random_state)
