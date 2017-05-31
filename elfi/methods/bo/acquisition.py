import logging
import numpy as np

from scipy.stats import uniform, multivariate_normal

from elfi.methods.bo.utils import minimize


logger = logging.getLogger(__name__)

# TODO: make a faster optimization method utilizing parallelization (see e.g. GPyOpt)


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
    priors : list of elfi.Priors, optional
        By default uniform distribution within model bounds.
    n_inits : int, optional
        Number of initialization points in internal optimization.
    max_opt_iters : int, optional
        Max iterations to optimize when finding the next point.
    noise_cov : float or np.array, optional
        Covariance of the added noise. If float, multiplied by identity matrix.
    seed : int
    """
    def __init__(self, model, priors=None, n_inits=10, max_opt_iters=1000, noise_cov=0., seed=0):
        self.model = model
        self.n_inits = n_inits
        self.max_opt_iters = int(max_opt_iters)

        if priors is None:
            self.priors = [None] * model.input_dim
        else:
            self.priors = priors

        if isinstance(noise_cov, (float, int)):
            noise_cov = np.eye(self.model.input_dim) * noise_cov
        self.noise_cov = noise_cov

        self.random_state = np.random.RandomState(seed)

    def evaluate(self, x, t=None):
        """Evaluates the acquisition function value at 'x'.

        Parameters
        ----------
        x : numpy.array
        t : int
            current iteration (starting from 0)
        """
        raise NotImplementedError

    def evaluate_grad(self, x, t=None):
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
        grad_obj = lambda x: self.evaluate_grad(x, t)
        minloc, minval = minimize(obj, grad_obj, self.model.bounds, self.priors, self.n_inits, self.max_opt_iters)
        x = np.tile(minloc, (n_values, 1))

        # add some noise for more efficient exploration
        x += multivariate_normal.rvs(cov=self.noise_cov, size=n_values, random_state=self.random_state) \
             .reshape((n_values, -1))

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

    def evaluate(self, x, t=None):
        """Lower confidence bound selection criterion: 
        
        mean - sqrt(\beta_t) * std
        
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        if not isinstance(t, int):
            raise ValueError("Parameter 't' should be an integer.")

        mean, var = self.model.predict(x)
        return mean - np.sqrt(self._beta(t) * var)

    def evaluate_grad(self, x, t=None):
        """Gradient of the lower confidence bound selection criterion.
        
        Parameters
        ----------
        x : numpy.array
        t : int
            Current iteration (starting from 0).
        """
        mean, var = self.model.predict(x)
        grad_mean, grad_var = self.model.predictive_gradients(x)
        grad_mean = grad_mean[:, :, 0]  # assume 1D output

        return grad_mean - 0.5 * grad_var * np.sqrt(self._beta(t) / var)


class UniformAcquisition(AcquisitionBase):

    def acquire(self, n_values, pending_locations=None, t=None):
        bounds = np.stack(self.model.bounds)
        return uniform(bounds[:,0], bounds[:,1] - bounds[:,0])\
            .rvs(size=(n_values, self.model.input_dim), random_state=self.random_state)
