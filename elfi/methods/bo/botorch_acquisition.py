# SPDX-FileCopyrightText: 2024 Yannick Kuhn <Yannick.Kuhn@dlr.de>
#
# SPDX-License-Identifier: BSD-3-Clause


from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.generation.gen import gen_candidates_torch, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
from elfi.methods.bo.acquisition import AcquisitionBase
from numpy import log, pi, random
from torch import float, long, tensor


class BoTorchAcquisitionBase(AcquisitionBase):
    """Mimics the AcquisitionBase in ELFI, but botorch-compatible.
    """

    def __init__(
        self,
        model,
        prior=None,
        n_inits=10,
        max_opt_iters=1000,
        noise_var=None,
        exploration_rate=10,
        seed=None,
        constraints=None,
    ):
        """
        :param model:
            An instance of ``GPyTorchRegression``. Needs to have the
            following attributes: `input_dim`, `bounds`,
            `parameter_names`, and a callable `predict(x, noiseless)`.
        """
        self.model = model
        if prior is not None:
            raise NotImplementedError(
                "Priors for acquisition function not implemented."
            )
        else:
            self.prior = prior
        self.n_inits = int(n_inits)
        self.max_opt_iters = int(max_opt_iters)
        self.constraints = constraints
        if noise_var is not None:
            self._check_noise_var(noise_var)
            self.noise_var = self._transform_noise_var(noise_var)
        else:
            self.noise_var = noise_var
        self.exploration_rate = exploration_rate
        self.random_state = (
            random if seed is None else random.RandomState(seed)
        )
        self.seed = 0 if seed is None else seed

    def acquire(self, n, t=None):
        raise NotImplementedError


class BoTorchLCBSC(BoTorchAcquisitionBase):
    def _beta(self, t):
        # Count from 0.
        if t is None:
            t = 0
        t += 1
        d = self.model.input_dim
        return 2 * log(t**(2 * d + 2) * pi**2 / (3 / self.exploration_rate))

    def evaluate(self, x, t=None):
        """
        Evaluate the acquisition function at `x`.

        :param x:
            The test variable to evaluate at.
        :param t:
            The index of the current iteration, starting from 0.
        :returns:
            the lower confindence bound at `x`.
        """
        lcb = UpperConfidenceBound(
            self.model._gp,
            beta=self._beta(t),
            maximize=False
        )
        return -lcb(x)

    def acquire(self, n, t=None, std_scale=None):
        """
        Minimize the acquisition function.

        :param n:
            Number of acquisition points to return.
        :param t:
            The index of the current iteration, starting from 0.
        :param std_scale:
            If left at None, logarithmically scales up the exploration.
            Else gives a fixed numerical value for the prefactor of the
            standard deviation in ``mu - sqrt(prefactor) * std``.
        :returns:
            np.ndarray; the shape is ``(n, input_dim)``.
        """
        lcb = UpperConfidenceBound(
            self.model._gp,
            beta=std_scale or self._beta(t),
            maximize=False,
        )
        bounds = self.model.torch_bounds
        x_init = gen_batch_initial_conditions(
            lcb,
            bounds,
            q=n,
            num_restarts=25,
            raw_samples=500 * 2**self.model.input_dim,
            inequality_constraints=[
                (tensor([i], dtype=long),
                 tensor([1], dtype=float),
                 b)
                for i, b in enumerate(bounds[0])
            ] + [
                (tensor([i], dtype=long),
                 tensor([-1], dtype=float),
                 -b)
                for i, b in enumerate(bounds[1])
            ]
        )
        # While gen_candidates_scipy is exact, the noise is actually useful.
        batch_candidates, batch_acq_values = gen_candidates_torch(
            initial_conditions=x_init,
            acquisition_function=lcb,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
        )
        candidate = get_best_candidates(batch_candidates, -batch_acq_values)
        return candidate.detach().numpy()


class SOBERUCB:
    def __init__(self, model, label="UCB", sample_size=1, exploration_rate=10):
        self.label = label
        self.exploration_rate = exploration_rate
        self.beta = (2 * log(
            sample_size**(2 * model.dim + 2) * pi**2
            / (3 / self.exploration_rate)
        ))
        self.af = UpperConfidenceBound(model, beta=self.beta)

    def __call__(self, x):
        return self.af(x.unsqueeze(1)).detach()
