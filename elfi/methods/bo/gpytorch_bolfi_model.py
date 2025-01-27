# SPDX-FileCopyrightText: 2024 Yannick Kuhn <Yannick.Kuhn@dlr.de>
#
# SPDX-License-Identifier: BSD-3-Clause

from botorch.models.gp_regression import SingleTaskGP
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means.mean import Mean
from gpytorch.priors import GammaPrior, NormalPrior
import numpy as np
from scipy.optimize import minimize
import torch


class ParabolicMean(Mean):
    """
    A parabolic Prior mean function, i.e.:

    µ(x) = a ⋅ x² + b ⋅ x + c

    or more generally for multidimensional inputs:

    µ(x) = Σⱼ (aⱼ ⋅ xⱼ² + bⱼ ⋅ xⱼ + c)
    """

    def __init__(
        self,
        list_of_square_priors,
        list_of_linear_priors,
        constant_prior,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        super().__init__()

        self.dim = len(list_of_square_priors)
        self.batch_shape = batch_shape

        for j, square_prior in enumerate(list_of_square_priors):
            self.register_parameter(
                name='raw_square_coefficient_' + str(j),
                parameter=torch.nn.Parameter(
                    square_prior.mean * torch.ones(batch_shape)
                )
            )
            self.register_prior(
                'square_coefficient_' + str(j) + '_prior',
                square_prior,
                lambda module, index=j: module.square_coefficients[index],
                lambda module, value, index=j: (
                    module._square_closure(value, index)
                ),
            )
            self.register_constraint(
                'raw_square_coefficient_' + str(j), Positive()
            )

        for j, linear_prior in enumerate(list_of_linear_priors):
            self.register_parameter(
                name='raw_linear_coefficient_' + str(j),
                parameter=torch.nn.Parameter(
                    linear_prior.mean * torch.ones(batch_shape)
                )
            )
            self.register_prior(
                'linear_coefficient_' + str(j) + '_prior',
                linear_prior,
                lambda module, index=j: module.linear_coefficients[index],
                lambda module, value, index=j: (
                    module._linear_closure(value, index)
                ),
            )

        self.register_parameter(
            name='raw_constant',
            parameter=torch.nn.Parameter(
                constant_prior.mean * torch.ones(batch_shape)
            )
        )
        self.register_prior(
            'constant_prior',
            constant_prior,
            lambda module: module.constant,
            lambda module, value: module._constant_closure(value)
        )

    @property
    def square_coefficients(self):
        return torch.stack([
            getattr(
                self, 'raw_square_coefficient_' + str(index) + '_constraint'
            ).transform(
                getattr(self, 'raw_square_coefficient_' + str(index))
            )
            for index in range(self.dim)
        ], dim=0)

    @square_coefficients.setter
    def square_coefficients(self, value):
        # When value is e.g. a NumPy array, convert to a Torch tensor
        # and copy dtype and device (CPU/GPU) from raw parameters.
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.square_coefficients)

        for index in range(self.dim):
            self._square_closure(self, value, index)

    def _square_closure(self, value, index):
        self.initialize(
            **{
                'raw_square_coefficient_' + str(index): (
                    getattr(
                        self,
                        'raw_square_coefficient_' + str(index) + '_constraint'
                    ).inverse_transform(value)
                )
            }
        )

    @property
    def linear_coefficients(self):
        return torch.stack([
            getattr(self, 'raw_linear_coefficient_' + str(index))
            for index in range(self.dim)
        ], dim=0)

    @linear_coefficients.setter
    def linear_coefficients(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.linear_coefficients)

        for index in range(self.dim):
            self._linear_closure(self, value, index)

    def _linear_closure(self, value, index):
        self.initialize(
            **{'raw_linear_coefficient_' + str(index): value}
        )

    @property
    def constant(self):
        return self.raw_constant

    @constant.setter
    def constant(self, value):
        self._constant_closure(self, value)

    def _constant_closure(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.constant)

        self.initialize(raw_constant=value)

    def forward(self, x):
        # The unsqueeze makes these have shape batch_shape x 1.
        square_coefficients = self.square_coefficients.unsqueeze(-1)
        linear_coefficients = self.linear_coefficients.unsqueeze(-1)
        constant = self.constant.unsqueeze(-1)

        square_term = torch.sum(square_coefficients.T * x**2, dim=-1)
        linear_term = torch.sum(linear_coefficients.T * x, dim=-1)

        return square_term + linear_term + constant


def BOLFIKernel(num_dims, length_scale, kernel_var):
    # In original BOLFI, a variance term ("bias") would be added.
    return ScaleKernel(
        # ProductStructureKernel(
        RBFKernel(
            lengthscale_prior=GammaPrior(length_scale, 1),
        ),
        #     num_dims=num_dims
        # ),
        outputscale_prior=GammaPrior(kernel_var, 1))


class BOLFIKernel_manually_implemented(gpytorch.kernels.Kernel):
    """
    Diagonal RBF kernel with importance weighting, i.e.:

    v(x, y) = σ_f² ⋅ exp((x - y)² / λ²)

    or more generally for multidimensional inputs:

    v(x, y) = σ_f² ⋅ exp(Σⱼ (xⱼ - yⱼ)² / λⱼ²)

    `list_of_lengthscale_priors` refers to the λⱼ.
    `multiplicative_variance` refers to σ_f².
    """

    is_stationary = True

    def __init__(
        self,
        list_of_lengthscale_priors,
        multiplicative_variance_prior,
        batch_shape=torch.Size(),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dim = len(list_of_lengthscale_priors)

        for j, lengthscale_prior in enumerate(list_of_lengthscale_priors):
            # Register the raw hyperparameter (i.e. without constraints)
            # with the inherited helper method register_parameter.
            self.register_parameter(
                name='raw_lengthscale_coefficient_' + str(j),
                parameter=torch.nn.Parameter(
                    torch.zeros(self.batch_shape)
                )
            )

            # Register the constraint with the inherited helper method
            # register_constraint.
            self.register_constraint(
                'raw_lengthscale_coefficient_' + str(j), Positive()
            )

            # Set the parameter Prior with the inherited helper method
            # register_prior.
            # The first argument of the lambdas is an instance of this
            # class, so treat it as self. The first lambda calls the
            # method of this class that defines the parameter; its
            # result will be input in the underlying probability
            # distribution, initialized by the second argument.
            # The second lambda additionally takes a tensor in
            # (transformed) parameter space and initializes the internal
            # parameter representation to the proper value by applying
            # the inverse transform.
            # This enables sampling parameter values from Priors.
            if lengthscale_prior is not None:
                self.register_prior(
                    'lengthscale_coefficient_' + str(j) + '_prior',
                    lengthscale_prior,
                    lambda module, index=j: (
                        module.lengthscale_coefficients[index]
                    ),
                    lambda module, value, index=j: (
                        module._lengthscale_closure(value, index)
                    ),
                )

        self.register_parameter(
            name='raw_multiplicative_variance',
            parameter=torch.nn.Parameter(torch.zeros(batch_shape))
        )
        self.register_constraint('raw_multiplicative_variance', Positive())
        self.register_prior(
            'multiplicative_variance_prior',
            multiplicative_variance_prior,
            lambda module: module.multiplicative_variance,
            lambda module, value: (
                module._multiplicative_variance_closure(value)
            )
        )
        self.register_parameter(
            name='raw_multiplicative_variance',
            parameter=torch.nn.Parameter(torch.zeros(batch_shape))
        )
        self.register_constraint('raw_multiplicative_variance', Positive())
        self.register_prior(
            'multiplicative_variance_prior',
            multiplicative_variance_prior,
            lambda module: module.multiplicative_variance,
            lambda module, value: (
                module._multiplicative_variance_closure(value)
            )
        )

    @property
    def lengthscale_coefficients(self):
        """Actual definition of the length scale as such."""
        # The attributes stem from the register_* helper methods.
        # Any constraints are applied here.
        return torch.stack([
            getattr(
                self,
                'raw_lengthscale_coefficient_' + str(index) + '_constraint'
            ).transform(
                getattr(self, 'raw_lengthscale_coefficient_' + str(index))
            )
            for index in range(self.dim)
        ], dim=0)

    @lengthscale_coefficients.setter
    def lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.square_coefficients)

        for index in range(self.dim):
            self._lengthscale_closure(self, value, index)

    def _lengthscale_closure(self, value, index):
        # When setting the paramater, transform the actual value to a raw one
        # by applying the inverse transform. initialize is inherited.
        self.initialize(
            **{
                'raw_lengthscale_coefficient_' + str(index): (
                    getattr(
                        self,
                        'raw_lengthscale_coefficient_' + str(index)
                        + '_constraint'
                    ).inverse_transform(value[index])
                )
            }
        )

    @property
    def multiplicative_variance(self):
        return self.raw_multiplicative_variance_constraint.transform(
            self.raw_multiplicative_variance
        )

    @multiplicative_variance.setter
    def multiplicative_variance(self, value):
        self._multiplicative_variance_closure(self, value)

    def _multiplicative_variance_closure(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.multiplicative_variance)

        self.initialize(raw_multiplicative_variance=value)

    def forward(self, x1, x2, **params):
        """Kernel function."""
        # Read: x_ = x / self.lengthscale.
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        # covar_dist is an inherited helper method for computing the
        # Euclidean distance between all pairs of points in x1 and x2.
        # diff_matrix = self.covar_dist(x1_, x2_, **params)
        diff_squared = torch.sum((x1_ - x2_)**2, dim=1)

        return self.multiplicative_variance * torch.exp(diff_squared)


class BOLFIModel(SingleTaskGP):

    def __init__(
        self,
        train_x,
        train_y,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(0)
        ),
        bounds=[]
    ):
        if bounds == []:
            raise NotImplementedError(
                "Default bounds not implemented yet, please provide them."
            )

        self.bounds = bounds

        self.dim = len(train_x[0])

        # Heuristics to choose kernel parameters based on the initial
        # data.
        length_scale = (torch.max(self.bounds) - torch.min(self.bounds)) / 3
        kernel_var = (torch.max(train_y) / 3)**2
        # This refers to the fallback in BOLFI when no mean function is
        # specified. Instead, a bias term gets added to the kernel.
        # The bias term used has a Gamma distribution with expected
        # value and variance both equal to bias_var.
        # bias_var = kernel_var / 4
        """
        # In BOLFI, all Hyperpriors in the kernel are Gamma
        # distributions. The lengthscale Priors are set to have expected
        # value and variance
        # both equal to length_scale. The multiplicate variance is set
        # to have expected value and variance both equal to kernel_var.
        self.covar_module = BOLFIKernel_manually_implemented(
            list_of_lengthscale_priors=[
                GammaPrior(length_scale, 1) for _ in range(self.dim)
            ],
            multiplicative_variance_prior=GammaPrior(kernel_var, 1),
            # bias_variance_prior=GammaPrior(bias_var, 1),
            # The first index in the shape is the "length" of the batch,
            # while the second is always 1 due to technical reasons.
            batch_shape=torch.Size([1, 1])
        )
        """
        covar_module = BOLFIKernel(self.dim, length_scale, kernel_var)

        # Use the initial parabolic fit to the initial training data
        # as means of the hyperparameters.
        def parabolic_fit_function(x, args):
            a = np.array(args[:self.dim])
            b = np.array(args[self.dim:-1])
            c = args[-1]
            return np.sum(a * x**2 + b * x, axis=1) + c

        np_x = np.array(train_x.cpu())
        # Since BoTorch requires an explicit output dimension, but it is
        # assumed to be 1 here, re-squeeze the training tensor.
        np_y = np.array(train_y.squeeze().cpu())

        if len(train_x) == 1:
            np_y = np.atleast_1d(np_y)
            a = np.zeros(self.dim)
            b = np.zeros(self.dim)
            c = np_y[0]
        elif len(train_x) == 2:
            a = np.zeros(self.dim)
            b = (np_y[1] - np_y[0]) / (np_x[1] - np_x[0])
            c = np_y[0] - np.sum(b * np_x[0])
        else:
            parabolic_fit = minimize(
                lambda *args: np.sum(
                    (parabolic_fit_function(np_x, *args) - np_y)**2
                )**0.5,
                x0=np.append(np.ones(self.dim), np.zeros(self.dim + 1)),
                method='trust-constr'
            ).x
            a = np.array(parabolic_fit[:self.dim])
            b = np.array(parabolic_fit[self.dim:-1])
            c = parabolic_fit[-1]

        # Heuristics for the variances of the parabolic hyperparameters.
        # a: check the "squaredness" of (train_y - b ⋅ train_x - c) / a.
        if len(train_x) > 2:
            a_var = np.atleast_1d(np.sum(
                (np.sqrt(np.abs(
                    (np_y[:, None] - b * np_x - c) / a
                )) - np_x)**2, axis=0
            ))  # **0.5 would be standard deviation
        else:
            a_var = np.ones(self.dim)
        # b: check the deviation of parabola minimum and data minimum
        # location.
        b_var = np.atleast_1d((-b - 2 * a * np_x[np.argmin(np_y)])**2)
        # c: check the deviation of parabola minimum and data minimum.
        c_var = (c - np.min(np_y))**2

        mean_module = ParabolicMean(
            [NormalPrior(a_j, a_j_var) for a_j, a_j_var in zip(a, a_var)],
            [NormalPrior(b_j, b_j_var) for b_j, b_j_var in zip(b, b_var)],
            NormalPrior(c, c_var),
            # The first index in the shape is the "length" of the batch,
            # while the second is always 1 due to technical reasons.
            # batch_shape=torch.Size([1, 1])
        )

        super().__init__(
            train_X=train_x,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            mean_module=mean_module
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
