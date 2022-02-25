"""This module contains an interface for using the GPy library in ELFI."""

# TODO: make own general GPRegression and kernel classes

import copy
import logging

import GPy
import numpy as np

logger = logging.getLogger(__name__)
logging.getLogger("GP").setLevel(logging.WARNING)  # GPy library logger


class GPyRegression:
    """Gaussian Process regression using the GPy library.

    GPy API: https://sheffieldml.github.io/GPy/
    """

    def __init__(self,
                 parameter_names=None,
                 bounds=None,
                 optimizer="scg",
                 max_opt_iters=50,
                 gp=None,
                 **gp_params):
        """Initialize GPyRegression.

        Parameters
        ----------
        parameter_names : list of str, optional
            Names of parameter nodes. If None, sets dimension to 1.
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters.
            `{'parameter_name':(lower, upper), ... }`
            If not supplied, defaults to (0, 1) bounds for all dimensions.
        optimizer : string, optional
            Optimizer for the GP hyper parameters
            Alternatives: "scg", "fmin_tnc", "simplex", "lbfgsb", "lbfgs", "sgd"
            See also: paramz.Model.optimize()
        max_opt_iters : int, optional
        gp : GPy.model.GPRegression instance, optional
        **gp_params
            kernel : GPy.Kern
            noise_var : float
            mean_function

        """
        if parameter_names is None:
            input_dim = 1
        elif isinstance(parameter_names, (list, tuple)):
            input_dim = len(parameter_names)
        else:
            raise ValueError("Keyword `parameter_names` must be a list of strings")

        if bounds is None:
            logger.warning('Parameter bounds not specified. Using [0,1] for each parameter.')
            bounds = [(0, 1)] * input_dim
        elif len(bounds) != input_dim:
            raise ValueError(
                'Length of `bounds` ({}) does not match the length of `parameter_names` ({}).'
                .format(len(bounds), input_dim))

        elif isinstance(bounds, dict):
            if len(bounds) == 1:  # might be the case parameter_names=None
                bounds = [bounds[n] for n in bounds.keys()]
            else:
                # turn bounds dict into a list in the same order as parameter_names
                bounds = [bounds[n] for n in parameter_names]
        else:
            raise ValueError("Keyword `bounds` must be a dictionary "
                             "`{'parameter_name': (lower, upper), ... }`")

        self.parameter_names = parameter_names
        self.input_dim = input_dim
        self.bounds = bounds

        self.gp_params = gp_params

        self.optimizer = optimizer
        self.max_opt_iters = max_opt_iters

        self._gp = gp

        self._rbf_is_cached = False
        self.is_sampling = False  # set to True once in sampling phase

    def __str__(self):
        """Return GPy's __str__."""
        return self._gp.__str__()

    def __repr__(self):
        """Return GPy's __str__."""
        return self.__str__()

    def predict(self, x, noiseless=False):
        """Return the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]
        noiseless : bool
            whether to include the noise variance or not to the returned variance

        Returns
        -------
        tuple
            GP (mean, var) at x where
                mean : np.array
                    with shape (x.shape[0], 1)
                var : np.array
                    with shape (x.shape[0], 1)

        """
        # Ensure it's 2d for GPy
        x = np.asanyarray(x).reshape((-1, self.input_dim))

        if self._gp is None:
            # TODO: return from GP mean function if given
            return np.zeros((x.shape[0], 1)), \
                np.ones((x.shape[0], 1))

        # direct (=faster) implementation for RBF kernel
        if self.is_sampling and self._kernel_is_default:
            if not self._rbf_is_cached:
                self._cache_RBF_kernel()

            r2 = np.sum(x**2., 1)[:, None] + self._rbf_x2sum - 2. * x.dot(self._gp.X.T)
            kx = self._rbf_var * np.exp(r2 * self._rbf_factor) + self._rbf_bias
            mu = kx.dot(self._rbf_woodbury)

            var = self._rbf_var + self._rbf_bias
            var -= kx.dot(self._rbf_woodbury_inv.dot(kx.T))
            var += self._rbf_noisevar  # likelihood

            return mu, var
        else:
            self._rbf_is_cached = False  # in case one resumes fitting the GP after sampling

        if noiseless:
            return self._gp.predict_noiseless(x)
        else:
            return self._gp.predict(x)

    # TODO: find a more general solution
    # cache some RBF-kernel-specific values for faster sampling
    def _cache_RBF_kernel(self):
        self._rbf_var = float(self._gp.kern.rbf.variance)
        self._rbf_factor = -0.5 / float(self._gp.kern.rbf.lengthscale)**2
        self._rbf_bias = float(self._gp.kern.bias.K(self._gp.X)[0, 0])
        self._rbf_noisevar = float(self._gp.likelihood.variance[0])
        self._rbf_woodbury = self._gp.posterior.woodbury_vector
        self._rbf_woodbury_inv = self._gp.posterior.woodbury_inv
        self._rbf_woodbury_chol = self._gp.posterior.woodbury_chol
        self._rbf_x2sum = np.sum(self._gp.X**2., 1)[None, :]
        self._rbf_is_cached = True

    def predict_mean(self, x):
        """Return the GP model mean function at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        np.array
            with shape (x.shape[0], 1)

        """
        return self.predict(x)[0]

    def predictive_gradients(self, x):
        """Return the gradients of the GP model mean and variance at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        tuple
            GP (grad_mean, grad_var) at x where
                grad_mean : np.array
                    with shape (x.shape[0], input_dim)
                grad_var : np.array
                    with shape (x.shape[0], input_dim)

        """
        # Ensure it's 2d for GPy
        x = np.asanyarray(x).reshape((-1, self.input_dim))

        if self._gp is None:
            # TODO: return from GP mean function if given
            return np.zeros((x.shape[0], self.input_dim)), \
                np.zeros((x.shape[0], self.input_dim))

        # direct (=faster) implementation for RBF kernel
        if self.is_sampling and self._kernel_is_default:
            if not self._rbf_is_cached:
                self._cache_RBF_kernel()

            r2 = np.sum(x**2., 1)[:, None] + self._rbf_x2sum - 2. * x.dot(self._gp.X.T)
            kx = self._rbf_var * np.exp(r2 * self._rbf_factor)
            dkdx = 2. * self._rbf_factor * (x - self._gp.X) * kx.T
            grad_mu = dkdx.T.dot(self._rbf_woodbury).T

            v = np.linalg.solve(self._rbf_woodbury_chol, kx.T + self._rbf_bias)
            dvdx = np.linalg.solve(self._rbf_woodbury_chol, dkdx)
            grad_var = -2. * dvdx.T.dot(v).T
        else:
            grad_mu, grad_var = self._gp.predictive_gradients(x)
            grad_mu = grad_mu[:, :, 0]  # Assume 1D output (distance in ABC)

        return grad_mu, grad_var

    def predictive_gradient_mean(self, x):
        """Return the gradient of the GP model mean at x.

        Parameters
        ----------
        x : np.array
            numpy compatible (n, input_dim) array of points to evaluate
            if len(x.shape) == 1 will be cast to 2D with x[None, :]

        Returns
        -------
        np.array
            with shape (x.shape[0], input_dim)

        """
        return self.predictive_gradients(x)[0]

    def _init_gp(self, x, y):
        self._kernel_is_default = False

        if self.gp_params.get('kernel') is None:
            kernel = self._default_kernel(x, y)

            if self.gp_params.get('noise_var') is None and self.gp_params.get(
                    'mean_function') is None:
                self._kernel_is_default = True

        else:
            kernel = self.gp_params.get('kernel')

        noise_var = self.gp_params.get('noise_var') or np.max(y)**2. / 100.
        mean_function = self.gp_params.get('mean_function')
        self._gp = self._make_gpy_instance(
            x, y, kernel=kernel, noise_var=noise_var, mean_function=mean_function)

    def _default_kernel(self, x, y):
        # Some heuristics to choose kernel parameters based on the initial data
        length_scale = (np.max(self.bounds) - np.min(self.bounds)) / 3.
        kernel_var = (np.max(y) / 3.)**2.
        bias_var = kernel_var / 4.

        # Construct a default kernel
        kernel = GPy.kern.RBF(input_dim=self.input_dim)

        # Set the priors
        kernel.lengthscale.set_prior(
            GPy.priors.Gamma.from_EV(length_scale, length_scale), warning=False)
        kernel.variance.set_prior(GPy.priors.Gamma.from_EV(kernel_var, kernel_var), warning=False)

        # If no mean function is specified, add a bias term to the kernel
        if 'mean_function' not in self.gp_params:
            bias = GPy.kern.Bias(input_dim=self.input_dim)
            bias.set_prior(GPy.priors.Gamma.from_EV(bias_var, bias_var), warning=False)
            kernel += bias

        return kernel

    def _make_gpy_instance(self, x, y, kernel, noise_var, mean_function):
        return GPy.models.GPRegression(
            X=x, Y=y, kernel=kernel, noise_var=noise_var, mean_function=mean_function)

    def update(self, x, y, optimize=False):
        """Update the GP model with new data.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize hyperparameters.

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
            # It seems that GPy will do some optimization unless you make copies of everything
            kernel = self._gp.kern.copy() if self._gp.kern else None
            noise_var = self._gp.Gaussian_noise.variance[0]
            mean_function = self._gp.mean_function.copy() if self._gp.mean_function else None
            self._gp = self._make_gpy_instance(
                x, y, kernel=kernel, noise_var=noise_var, mean_function=mean_function)

        if optimize:
            self.optimize()

    def optimize(self):
        """Optimize GP hyperparameters."""
        logger.debug("Optimizing GP hyperparameters")
        try:
            self._gp.optimize(self.optimizer, max_iters=self.max_opt_iters)
        except np.linalg.linalg.LinAlgError:
            logger.warning("Numerical error in GP optimization. Stopping optimization")

    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        if self._gp is None:
            return 0
        return self._gp.num_data

    @property
    def X(self):
        """Return input evidence."""
        return self._gp.X

    @property
    def Y(self):
        """Return output evidence."""
        return self._gp.Y

    @property
    def noise(self):
        """Return the noise."""
        return self._gp.Gaussian_noise.variance[0]

    @property
    def instance(self):
        """Return the gp instance."""
        return self._gp

    def copy(self):
        """Return a copy of current instance."""
        kopy = copy.copy(self)
        if self._gp:
            kopy._gp = self._gp.copy()

        if 'kernel' in self.gp_params:
            kopy.gp_params['kernel'] = self.gp_params['kernel'].copy()

        if 'mean_function' in self.gp_params:
            kopy.gp_params['mean_function'] = self.gp_params['mean_function'].copy()

        return kopy

    def __copy__(self):
        """Return a copy of current instance."""
        return self.copy()
