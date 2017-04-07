# TODO: rename file to GPyRegression
# TODO: make own general GPRegression and kernel classes

import logging
import numpy as np
import copy
import GPy

logger = logging.getLogger(__name__)
logging.getLogger("GP").setLevel(logging.WARNING)  # GPy library logger


class GPyRegression:
    """Gaussian Process regression using the GPy library.

    GPy API: https://sheffieldml.github.io/GPy/

    Parameters
    ----------
    
    input_dim : int
        number of input dimensions
    bounds : tuple of (min, max) tuples
        Input space box constraints as a tuple of pairs, one for each input dim
        Eg: ((0, 1), (0, 2), (-2, 2))
        If not supplied, defaults to (0, 1) bounds for all dimenstions.
    optimizer : string
            Optimizer for the GP hyper parameters
            Alternatives: "scg", "fmin_tnc", "simplex", "lbfgsb", "lbfgs", "sgd"
            See also: paramz.Model.optimize()
    max_opt_iters : int
    gp : GPy.model.GPRegression instance
    **gp_params
        kernel : GPy.Kern
        noise_var : float
        mean_function

    """

    def __init__(self, input_dim=None, bounds=None, optimizer="scg", max_opt_iters=50,
                 gp=None, **gp_params):

        if not input_dim and not bounds:
            input_dim = 1

        if not input_dim:
            input_dim = len(bounds)

        if not bounds:
            logger.warning('Parameter bounds not specified. Using [0,1] for each '
                           'parameter.')
            bounds = [(0,1)] * input_dim

        if len(bounds) != input_dim:
            raise ValueError("Number of bounds({}) does not match input dimension ({})."
                             .format(input_dim, len(bounds)))

        self.input_dim = input_dim
        self.bounds = bounds

        self.gp_params = gp_params

        self.optimizer = optimizer
        self.max_opt_iters = max_opt_iters

        self._gp = gp

    def predict(self, x, noiseless=False):
        """Returns the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy (n, input_dim) array of points to evaluate
        noiseless : bool
            whether to include the noise variance or not to the returned variance
        
        Returns
        -------
        tuple
            GP (mean, var) at x where
                mean : np.array
                    with shape (len(x), input_dim)
                var : np.array
                    with shape (len(x), input_dim)
        """
        if self._gp is None:
            # TODO: return from GP mean function if given
            return np.zeros(len(x), self.input_dim), \
                   np.ones(len(x), self.input_dim)

        # Need to cast as 2d array for GPy
        x = x.reshape((-1, self.input_dim))
        if noiseless:
            return self._gp.predict_noiseless(x)
        else:
            return self._gp.predict(x)

    def predict_mean(self, x):
        """Returns the GP model mean function at x.
        """
        return self.predict(x)[0]

    def _init_gp(self, x, y):
        kernel = self.gp_params.get('kernel') or self._default_kernel(x, y)
        noise_var = self.gp_params.get('noise_var') or 1.
        mean_function = self.gp_params.get('mean_function')
        self._gp = self._make_gpy_instance(x, y, kernel=kernel, noise_var=noise_var,
                                           mean_function=mean_function)

    def _default_kernel(self, x, y):
        # TODO: add heuristics based on the initial data
        length_scale = 1.
        kernel_var = 1.
        bias_var = 1.

        # Priors
        # kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,100.), warning=False)
        # kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,100.), warning=False)
        # likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,100.), warning=False)

        # Construct a default kernel
        kernel = GPy.kern.RBF(input_dim=self.input_dim, variance=kernel_var,
                           lengthscale=length_scale)

        # If no mean function is specified, add a bias term to the kernel
        if 'mean_function' not in self.gp_params:
            kernel += GPy.kern.Bias(input_dim=self.input_dim, variance=bias_var)

        return kernel

    def _make_gpy_instance(self, x, y, kernel, noise_var, mean_function):
        return GPy.models.GPRegression(X=x, Y=y, kernel=kernel, noise_var=noise_var,
                                       mean_function=mean_function)

    def update(self, x, y, optimize=False):
        """Updates the GP model with new data
        """

        # Must cast these as 2d for GPy
        x = x.reshape((-1, self.input_dim))
        y = y.reshape((-1, 1))

        if self._gp is None:
            self._init_gp(x, y)
        else:
            # Reconstruct with new data
            x = np.r_[self._gp.X, x]
            y = np.r_[self._gp.Y, y]
            kernel = self._gp.kern
            noise_var = self._gp.Gaussian_noise.variance[0]
            mean_function = self._gp.mean_function
            self._gp = self._make_gpy_instance(x, y, kernel=kernel, noise_var=noise_var,
                                               mean_function=mean_function)

        if optimize:
            self.optimize()

    def optimize(self):
        """Optimize GP hyper parameters.
        """
        logger.debug("Optimizing GP hyper parameters")
        try:
            self._gp.optimize(self.optimizer, max_iters=self.max_opt_iters)
        except np.linalg.linalg.LinAlgError:
            logger.warning("Numerical error in GP optimization. Stopping optimization")

    @property
    def n_evidence(self):
        """Returns the number of observed samples.
        """
        if self._gp is None:
            return 0
        return self._gp.num_data

    def copy(self):
        kopy = copy.copy(self)
        if self._gp:
            kopy._gp = self._gp.copy()

        if 'kernel' in self.gp_params:
            kopy.gp_params['kernel'] = self.gp_params['kernel'].copy()

        if 'mean_function' in self.gp_params:
            kopy.gp_params['mean_function'] = self.gp_params['mean_function'].copy()

        return kopy


