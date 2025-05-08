# SPDX-FileCopyrightText: 2024 Yannick Kuhn <Yannick.Kuhn@dlr.de>
#
# SPDX-License-Identifier: BSD-3-Clause

# This module contains an interface for using GPyTorch in ELFI.

import copy
import gpytorch
import torch
import warnings

from elfi.methods.bo.gpytorch_bolfi_model import BOLFIModel


class GPyTorchRegression:
    """Gaussian Process regression using the GPyTorch library."""

    def __init__(
        self,
        parameter_names=None,
        bounds=None,
        optimizer=None,
        max_opt_iters=50,
        gp=None,
        initial_evidence=None,
        **gp_params
    ):
        """
        Initialize GPyTorchRegression.

        :param parameter_names: list of str, optional
            Names of parameter nodes. If None, sets dimension to 1.
        :param bounds: dict, optional
            The region where to estimate the posterior for each
            parameter in ``model.parameters``.
            ``{'parameter_name':(lower, upper), ... }``
            If not supplied, defaults to (0, 1) bounds for all
            dimensions.
        :param optimizer: string, optional
            Optimizer for the GP hyper parameters.
        :param max_opt_iters: int, optional
        :param gp: gpytorch.models.ExactGP, optional
        """

        if isinstance(initial_evidence, dict):
            raise ValueError(
                "Batch dict for initial evidence not implemented in this "
                "re-implementation of BOLFI. Please implement it like in ELFI."
            )

        if parameter_names is None:
            input_dim = 1
        elif isinstance(parameter_names, (list, tuple)):
            input_dim = len(parameter_names)
        else:
            raise ValueError(
                "Keyword `parameter_names` must be a list of strings."
            )

        if bounds is None:
            warnings.warn(
                "Parameter bounds not specified. Using [0,1] for each "
                "parameter."
            )
            bounds = [(0, 1)] * input_dim
        elif len(bounds) != input_dim:
            raise ValueError(
                (
                    "Length of `bounds` ({}) does not match the length of "
                    + "`parameter_names` ({})."
                ).format(len(bounds), input_dim)
            )
        elif isinstance(bounds, dict):
            if len(bounds) == 1:  # might be parameter_names==None
                bounds = [bounds[n] for n in bounds.keys()]
            else:
                # turn bounds dict into a list in the same order as
                # parameter_names
                bounds = [bounds[n] for n in parameter_names]
        else:
            raise ValueError(
                "Keyword `bounds` must be a dictionary "
                "`{'parameter_name': (lower, upper), ... }`"
            )

        # Make a 2 x d tensor of bounds for PyTorch and BoTorch.
        torch_bounds = torch.tensor(bounds).T

        self.parameter_names = parameter_names
        self.bounds = bounds
        self.torch_bounds = torch_bounds
        self.optimizer = optimizer
        self.max_opt_iters = max_opt_iters
        self._gp = gp
        self.gp_params = gp_params

        self.input_dim = input_dim
        self.is_sampling = False  # set to True once in sampling phase

        self.initial_evidence = initial_evidence
        self.initial_X = None
        self.initial_Y = None

        self.first_update = False

    # def __str__(self):
    #     """Return __str__ of underlying Gaussian Process."""
    #     return self._gp.__str__()

    # def __repr__(self):
    #     """Return __repr__ of underlying Gaussian Process."""
    #     return self._gp.__repr__()

    def predict(self, x, noiseless=True, numpy_output=True):
        """
        Return the GP model posterior mean and variance at x.

        Fast variance inference is made with LOVE via fast_pred_var().
        For accurate variance inference, you can just comment out the
        part.

        :param x: np.array
            NumPy compatible (n, input_dim) array of points to evaluate
            if ``len(x.shape) == 1`` will be cast to 2D with
            ``x[None, :]``.
        :param noiseless: bool
            Whether to include the noise variance or not to the returned
            variance.
        :param numpy_output: bool
            Compatibility with ELFI built-in methods. Set to False to
            retain PyTorch Tensor output.

        :returns:
            Returns as expected by ELFI:
            - pred.mean; torch.tensor, the predictive mean
            - pred.variance; torch.tensor, the predictive variance
            Shape:
            GP posterior (mean, var) at x where
                mean : np.array
                    with shape ``(x.shape[0], 1)``
                var : np.array
                    with shape ``(x.shape[0], 1)``
        """

        # Make sure that the GP is in "computing predictions" mode.
        self._gp.eval()
        self._gp.likelihood.eval()

        x = torch.Tensor(x)

        if not noiseless:
            raise NotImplementedError(
                "GP prediction with additional diagonal kernel noise not "
                "implemented."
            )

        try:
            with gpytorch.settings.fast_pred_var():
                pred = self._gp.likelihood(self._gp(x))
        except RuntimeError:
            warnings.warn("Cholesky failed. Adding more jitter...")
            with (
                gpytorch.settings.cholesky_jitter(float=1e-2)
            ):
                pred = self._gp.likelihood(self._gp(x))

        if numpy_output:
            return pred.mean.detach().numpy(), pred.variance.detach().numpy()
        else:
            return pred.mean, pred.variance

    def predict_mean(self, x, numpy_output=True):
        """
        Return the GP model posterior mean function at x.

        :param numpy_output: bool
            Compatibility with ELFI built-in methods. Set to False to
            retain PyTorch Tensor output.
        :returns:
            np.array with shape ``(x.shape[0], 1)``.
        """

        return self.predict(x, numpy_output=numpy_output)[0]

    def predictive_gradients(self, x, numpy_output=True):
        """
        Return the gradients of the GP model posterior mean and variance.

        Given a set of points at which to predict X* (size [N*,Q]),
        compute the derivatives of the mean and variance.
        Resulting arrays are sized:
            dmu_dX* -- [N*, Q ,D], where D is the number of output in
            this GP (usually one).

        Note that this is not the same as computing the mean and
        variance of the derivative of the function!
            dv_dX*  -- [N*, Q], since all outputs have same variance.

        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q) ]

        :param x: np.array
            NumPy compatible (n, input_dim) array of points to evaluate
            if ``len(x.shape) == 1`` will be cast to 2D with
            ``x[None, :]``.
        :param numpy_output: bool
            Compatibility with ELFI built-in methods. Set to False to
            retain PyTorch Tensor output.
        :returns:
            tuple
                GP (grad_mean, grad_var) at x where
                    grad_mean : np.array
                        with shape ``(x.shape[0], input_dim)``
                    grad_var : np.array
                        with shape ``(x.shape[0], input_dim)``
        """

        self._gp.eval()
        self._gp.likelihood.eval()

        def pred_evaluation(X_eval):
            try:
                with gpytorch.settings.fast_pred_var():
                    pred = self._gp.likelihood(self._gp(X_eval))
            except RuntimeError:
                warnings.warn("Cholesky failed. Adding more jitter...")
                with gpytorch.settings.cholesky.jitter(float=1e-2):
                    pred = self._gp.likelihood(self._gp(X_eval))
            return pred

        # Note how the gradient is stored in X_mean rather than the
        # function.
        X_mean = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
        pred_for_mean = pred_evaluation(X_mean)
        pred_mean_grad_term = pred_for_mean.mean.sum()
        pred_mean_grad_term.backward()
        pred_mean_grad = X_mean.grad

        X_var = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
        pred_for_var = pred_evaluation(X_var)
        pred_var_grad_term = pred_for_var.variance.sum()
        pred_var_grad_term.backward()
        pred_var_grad = X_var.grad

        if numpy_output:
            return (
                pred_mean_grad.detach().numpy(), pred_var_grad.detach().numpy()
            )
        else:
            return pred_mean_grad, pred_var_grad

    def predictive_gradient_mean(self, x, numpy_output=True):
        """
        Return the gradient of the GP model posterior mean at x.


        :param x: np.array
            NumPy compatible (n, input_dim) array of points to evaluate
            if ``len(x.shape) == 1`` will be cast to 2D with
            ``x[None, :]``.
        :param numpy_output: bool
            Compatibility with ELFI built-in methods. Set to False to
            retain PyTorch Tensor output.
        :returns:
            np.array with shape ``(x.shape[0], input_dim)``.
        """

        return self.predictive_gradients(x, numpy_output=numpy_output)[0]

    def update(self, x, y, optimize=False):
        """
        Update the GP model with new data.

        :param inputs:
            A torch.Tensor of the shape
            "b1 x ... x bk x m x d"
            or
            "f x b1 x ... x bk x m x d".
            Locations of (fantasy) observations.
        :param targets:
            A torch:Tensor of the shape
            "b1 x ... x bk x m"
            or
            "f x b1 x ... x bk x m".
            Labels of (fantasy) observations.
        :param optimize:
            Whether or not to optimize hyperparameters.
        :returns:
            An ``ExactGP`` model with ``n + m`` training examples, where
            the ``m`` fantasy examples have been added and all test-time
            caches have been updated.
        """

        x = torch.tensor(x, dtype=torch.double)
        y = torch.tensor(y, dtype=torch.double).unsqueeze(-1)
        initial_optimize = self.first_update

        if self.initial_X is None:
            self.initial_X = x
            self.initial_Y = y
            self._gp = BOLFIModel(
                self.initial_X, self.initial_Y, bounds=self.bounds
            )
            self._gp.set_train_data(self.X, self.Y, strict=False)
        elif self.initial_evidence > len(self.initial_X):
            self.initial_X = torch.cat((self.initial_X, x))
            self.initial_Y = torch.cat((self.initial_Y, y))
            self._gp = BOLFIModel(
                self.initial_X, self.initial_Y, bounds=self.bounds
            )
            self._gp.set_train_data(self.X, self.Y, strict=False)
            # Use two passes (the optimizer gets stored between samples.
            if len(self.initial_X) >= self.initial_evidence - 2:
                optimize = True
        else:
            # "get_fantasy_model" updates the GP, but discards previous
            # hyperparameter training, as the model gets replaced and
            # hence the links get broken.
            # self._gp = self._gp.get_fantasy_model(x[0], y[0])
            self._gp.set_train_data(
                torch.concatenate((self.X, x)),
                torch.concatenate((self.Y, y[0])),
                strict=False,
            )

        if optimize or initial_optimize:
            self.optimize()
            self.first_update = False

    def optimize(self):
        """Optimize GP hyperparameters. Use Adam by default."""
        self._gp.train()
        self._gp.likelihood.train()
        optimizer = self.optimizer if self.optimizer else torch.optim.Adam(
            self._gp.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters
        # One possible loss function for GPs: marginal log likehood.
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self._gp.likelihood, self._gp
        )
        for i in range(self.max_opt_iters):
            # Delete gradients from previous iteration by overwriting with 0.
            optimizer.zero_grad()
            # Output from model.
            output = self._gp(self.X)
            # Calculate loss and backpropagation gradients.
            loss = -mll(output, self.Y)
            loss.backward()
            # print(
            #     'Iter %d/%d - Loss: %.3f' % (
            #         i + 1, self.max_opt_iters, loss.item(),
            #     )
            # )
            optimizer.step()

    @property
    def n_evidence(self):
        """Return the number of observed samples."""
        if self._gp is None:
            return 0
        return len(self._gp.train_inputs[0])

    @property
    def X(self):
        """Return input evidence."""
        return self._gp.train_inputs[0]

    @property
    def Y(self):
        """Return output evidence."""
        return self._gp.train_targets

    @property
    def noise(self):
        """Return the noise."""
        # In GPy:
        # return self._gp.Gaussian_noise.variance[0]
        raise NotImplementedError(
            "No additional noise to GP implemented."
        )

    @property
    def instance(self):
        """Return the gp instance."""
        return self._gp

    def copy(self):
        """Return a copy of current instance."""
        kopy = copy.copy(self)
        if self._gp:
            kopy._gp = self._gp.copy()

        # if 'kernel' in self.gp_params:
        #     kopy.gp_params['kernel'] = self.gp_params['kernel'].copy()

        # if 'mean_function' in self.gp_params:
        #     kopy.gp_params['mean_function'] = (
        #         self.gp_params['mean_function'].copy()
        #     )

        return kopy

    def __copy__(self):
        """Return a copy of current instance."""
        return self.copy()
