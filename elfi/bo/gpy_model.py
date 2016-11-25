import logging
import numpy as np
import copy
import GPy

logger = logging.getLogger(__name__)
logging.getLogger("GP").setLevel(logging.WARNING)  # GPy library logger

class GPyModel():
    """Gaussian Process regression model using the GPy library implementation.

    GPy API: https://sheffieldml.github.io/GPy/

    Parameters
    ----------
    input_dim : int
        number of input dimensions
    bounds : tuple of (min, max) tuples
        Input space box constraints as a tuple of pairs, one for each input dim
        Eg: ((0, 1), (0, 2), (-2, 2))
        If not supplied, defaults to (0, 1) bounds for all dimenstions.
    kernel : GPy.kern kernel
        GPy compatible kernel function
        if not None, then the other kernel_* params are ignored
    kernel_class : GPy.kern classname
        type of kernel from GPy internal kernels
    kernel_var : float
        variance of kernel
    kernel_scale : float
        lengthscale of kernel
    noise_var : float
        observation noise variance
    optimizer : string
        Optimizer to use for adjusting model parameters.
        Alternatives: "scg", "fmin_tnc", "simplex", "lbfgsb", "lbfgs", "sgd"
        See also: paramz.Model.optimize()
    max_opt_iters : int
        Number of optimization iterations to run after each observed sample.


    Possible TODOs:
    - allow initialization with samples, which give hints to initial kernel params
    - allow giving GP object as parameter
    - priors for the GP
    - allow kernel bias term

    """

    def __init__(self, input_dim=1, bounds=None, kernel=None,
                 kernel_class=GPy.kern.RBF, kernel_var=1.0, kernel_scale=1.,
                 noise_var=0.5, optimizer="scg", max_opt_iters=50):
        self.input_dim = input_dim
        if self.input_dim < 1:
            raise ValueError("Input dimension needs to be larger than 1. " +
                    "Received {}.".format(input_dim))
        if bounds is not None:
            self.bounds = bounds
        else:
            logger.info("{}: No bounds supplied, defaulting to [0,1] bounds."
                    .format(self.__class__.__name__))
            self.bounds = [(0,1)] * self.input_dim
        if len(self.bounds) != self.input_dim:
            raise ValueError("Number of bounds should match input dimension. " +
                    "Expected {}. Received {}.".format(self.input_dim, len(self_bounds)))
        self.noise_var = noise_var
        self.optimizer = optimizer
        self.max_opt_iters = max_opt_iters
        self.gp = None
        self.set_kernel(kernel, kernel_class, kernel_var, kernel_scale)

    def evaluate(self, x):
        """Returns the GP model mean, variance and std at x.

        Parameters
        ----------
        x : numpy 1D array
            location to evaluate at

        Returns
        -------
        gp (mean, s2, s) at x : (float, float, float)
        """
        if self.gp is None:
            # TODO: return from GP prior
            return 0.0, 0.0, 0.0
        m, s2 = self.gp.predict(np.atleast_2d(x))
        if m != m:
            logger.warning("{}: Mean evaluated to '%s'."
                    .format(self.__class__.__name__, m))
        return float(m), float(s2), np.sqrt(float(s2))

    def eval_mean(self, x):
        """Returns the GP model mean function at x.

        Parameters
        ----------
        x : numpy 1d array
            location to evaluate at

        Returns
        -------
        gp mean value at x : float
        """
        m, s2, s = self.evaluate(x)
        return m

    def set_noise_var(self, noise_var=0.0):
        """Change GP observation noise variance and re-fit the GP.

        Parameters
        ----------
        see constructor
        """
        self.noise_var = noise_var
        if self.gp is not None:
            # re-fit gp with new noise variance
            self._fit_gp(self.gp.X, self.gp.Y)

    def set_kernel(self, kernel=None, kernel_class=None, kernel_var=None,
                   kernel_scale=None):
        """Changes the GP kernel to supplied and re-fit the GP.

        Parameters
        ----------
        see constructor
        """
        if kernel is not None:
            # explicit kernel supplied
            if kernel.input_dim != self.input_dim:
                raise ValueError("Kernel input_dim must match model input_dim.")
            self.kernel = kernel
        else:
            self.kernel_class = kernel_class or self.kernel_class
            self.kernel_var = kernel_var or self.kernel_var
            self.kernel_scale = kernel_scale or self.kernel_scale
            if isinstance(self.kernel_class, str):
                self.kernel_class = getattr(GPy.kern, self.kernel_class)
            self.kernel = self.kernel_class(input_dim=self.input_dim,
                                            variance=self.kernel_var,
                                            lengthscale=self.kernel_scale)
        if self.gp is not None:
            # re-fit gp with new kernel
            self._fit_gp(self.gp.X, self.gp.Y)

    def _fit_gp(self, X, Y):
        """Constructs the gp model.
        """
        self.gp = GPy.models.GPRegression(X=X, Y=Y,
                                          kernel=self.kernel,
                                          noise_var=self.noise_var)

        # FIXME: move to initialization
        self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,100.))
        self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,100.))
        self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,100.))

    def _within_bounds(self, x):
        """Returns true if location x is within model bounds.
        """
        for i, xi in enumerate(x):
            if not self.bounds[i][0] <= xi <= self.bounds[i][1]:
                return False
        return True

    def _check_input(self, X, Y):
        """Validates if input X and Y are acceptable.

        Raises a ValueError in case input is not acceptable.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Type of X must be numpy.ndarray. " +
                    "Received type {}.".format(type(X)))
        if not isinstance(Y, np.ndarray):
            raise ValueError("Type of Y must be numpy.ndarray. " +
                    "Received type {}.".format(type(Y)))
        if len(X.shape) != 2 or X.shape[1] != self.input_dim:
            raise ValueError("Shape of X must be (n_obs, {}). ".format(self.input_dim) +
                    "Received shape {}.".format(X.shape))
        if len(Y.shape) != 2 or Y.shape[1] != 1:
            raise ValueError("Shape of Y must be (n_obs, {}). ".format(self.input_dim) +
                    "Received shape {}.".format(X.shape))
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must contain equal number of observations " +
                    "(X.shape[0]={}, Y.shape[0]={}).".format(X.shape[0], Y.shape[0]))
        for x in X:
            if not self._within_bounds(x):
                raise ValueError("Location {} was not within model bounds.".format(x))
        return X, Y

    def update(self, X, Y):
        """Add (X, Y) as observations, updates GP model.

        Parameters
        ----------
        X : numpy 2D array
            observation locations, shape (n_obs, input_dim)
        Y : numpy 2D array
            observation values, shape (n_obs, 1)
        """
        self._check_input(X, Y)
        logger.debug("{}: Observed: %s at %s."
                    .format(self.__class__.__name__, X, Y))
        if self.gp is not None:
            X = np.vstack((self.gp.X, X))
            Y = np.vstack((self.gp.Y, Y))
        self._fit_gp(X, Y)
        self.optimize()

    def optimize(self, max_opt_iters=None, fail_on_error=False):
        """Optimize GP kernel parameters.

        Parameters
        ----------
        max_opt_iters : int or None
            Maximum number of optimization iterations.
            If None, will use self.max_opt_iters.
        fail_on_error : bool
            If False, will try to continue function in case
            a numerical error takes place in optimization.
        """
        if self.gp is None:
            return
        if max_opt_iters is None:
            max_opt_iters = self.max_opt_iters
        if max_opt_iters < 1:
            return
        try:
            self.gp.optimize(self.optimizer, max_iters=max_opt_iters)
        except np.linalg.linalg.LinAlgError:
            logger.warning("{}: Numerical error in GP optimization. Attempting to continue."
                    .format(self.__class__.__name__))
            if fail_on_error is True:
                raise

    @property
    def n_observations(self):
        """Returns the number of observed samples.
        """
        if self.gp is None:
            return 0
        return self.gp.num_data

    def copy(self):
        model = GPyModel(input_dim=self.input_dim,
                         bounds=self.bounds[:],
                         kernel=self.kernel.copy(),
                         noise_var=self.noise_var,
                         optimizer=self.optimizer,
                         max_opt_iters=self.max_opt_iters)
        if self.gp is not None:
            model._fit_gp(self.gp.X[:], self.gp.Y[:])
        return model

