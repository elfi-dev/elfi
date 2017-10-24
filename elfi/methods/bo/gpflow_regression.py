"""This module contains an interface for using the GPflow library in ELFI."""

# TODO: make own general GPRegression and kernel classes

import copy
import logging

import numpy as np

import gpflow

logger = logging.getLogger(__name__)
logging.getLogger("GP").setLevel(logging.WARNING)  # GPy library logger


class GPflowRegression:
    """Gaussian Process regression using the GPflow library.

    GPflow API: https://github.io/GPflow/
    """

    def __init__(self,
                 parameter_names=None,
                 bounds=None,
                 optimizer="L-BFGS-B",
                 max_opt_iters=50,
                 gp=None,
                 **gp_params):
        """Initialize GPflowRegression.

        Parameters
        ----------
        parameter_names : list of str, optional
            iernel = self._gp.kern.copy() if self._gp.kern else None
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
        gp : GPflow.gpr.GPR instance, optional
        **gp_params
            kernel : GPflow.kernels
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

        self.input_dim = input_dim
        self.bounds = bounds

        self.gp_params = gp_params

        self.optimizer = optimizer
        self.max_opt_iters = max_opt_iters

        self._gp = gp

        self._rbf_is_cached = False
        self.is_sampling = False  # set to True once in sampling phase

    def __str__(self):
        """Return GPflow's __str__."""
        return self._gp.__str__()

    def __repr__(self):
        """Return GPflow's __str__."""
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
        # Ensure it's 2d for GPflow
        x = np.asanyarray(x).reshape((-1, self.input_dim))
        if self._gp is None:
            # TODO: return from GP mean function if given
            return np.zeros((x.shape[0], 1)), \
                np.ones((x.shape[0], 1))
        if noiseless:
            return self._gp.predict_f(x)
        else:
            return self._gp.predict_y(x)

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
        # Ensure it's 2d for GPflow
        x = np.asanyarray(x).reshape((-1, self.input_dim))

        if self._gp is None:
            # TODO: return from GP mean function if given
            return np.zeros((x.shape[0], self.input_dim)), \
                np.zeros((x.shape[0], self.input_dim))

        grad_mu, grad_var = self._gp.predict_f_gradients(x)

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
        self._gp = self._make_gpflow_instance(
            x, y, kernel=kernel, mean_function=mean_function)

        self._gp.likelihood.variance = noise_var

    def _default_kernel(self, x, y):
        # Some heuristics to choose kernel parameters based on the initial data
        length_scale = (np.max(self.bounds) - np.min(self.bounds)) / 3.
        kernel_var = (np.max(y) / 3.)**2.
        bias_var = kernel_var / 4.

        # Construct a default kernel
        kernel = gpflow.kernels.RBF(input_dim=self.input_dim)

        # Set the priors
        kernel.lengthscales.prior = gpflow.priors.Gamma(length_scale, length_scale)
        kernel.variance.prior = gpflow.priors.Gamma(kernel_var, kernel_var)

        # If no mean function is specified, add a bias term to the kernel
        if 'mean_function' not in self.gp_params:
            bias = gpflow.kernels.Bias(input_dim=self.input_dim)
            bias.variance.prior = gpflow.priors.Gamma(bias_var, bias_var)
            kernel += bias

        return kernel

    def _make_gpflow_instance(self, x, y, kernel,  mean_function):
        return gpflow.gpr.GPR(
            X=x, Y=y, kern=kernel, mean_function=mean_function)

    def update(self, x, y, optimize=False):
        """Update the GP model with new data.

        Parameters
        ----------
        x : np.array
        y : np.array
        optimize : bool, optional
            Whether to optimize hyperparameters.

        """
        # Must cast these as 2d for GPflow
        x = x.reshape((-1, self.input_dim))
        y = y.reshape((-1, 1))

        if self._gp is None:
            self._init_gp(x, y)
        else:
            # Reconstruct with new data
            x = np.concatenate((self._gp.X.value, x), axis=0)
            y = np.concatenate((self._gp.Y.value, y), axis=0)
            noise_var = self._gp.likelihood.variance
            self._gp = self._make_gpflow_instance(
                x, y, kernel=self._gp.kern, mean_function=self._gp.mean_function)

            self._gp.likelihood.variance = noise_var

        if optimize:
            self.optimize()

    def optimize(self):
        """Optimize GP hyperparameters."""
        logger.debug("Optimizing GP hyperparameters")
        try:
            self._gp.optimize(self.optimizer, maxiter=self.max_opt_iters)
        except np.linalg.linalg.LinAlgError:
            logger.warning("Numerical error in GP optimization. Stopping optimization")

    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        if self._gp is None:
            return 0
        return self._gp.X.value.shape[0]

    @property
    def X(self):
        """Return input evidence."""
        return self._gp.X.value

    @property
    def Y(self):
        """Return output evidence."""
        return self._gp.Y.value

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
