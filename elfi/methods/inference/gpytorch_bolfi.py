# SPDX-FileCopyrightText: 2024 Yannick Kuhn <Yannick.Kuhn@dlr.de>
#
# SPDX-License-Identifier: BSD-3-Clause

from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.generation.gen import gen_candidates_scipy, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
from elfi.loader import get_sub_seed
from elfi.methods.inference.bolfi import BayesianOptimization, BOLFI
import elfi.methods.mcmc as mcmc
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.results import BolfiSample, OptimizationResult
from elfi.methods.utils import arr2d_to_batch  # , resolve_sigmas
from elfi.model.elfi_model import NodeReference
from elfi.model.extensions import ModelPrior
import gpytorch
import numpy as np
import pyro
from pyro.infer.mcmc import NUTS, MCMC
import torch

from elfi.methods.bo.botorch_acquisition import BoTorchLCBSC
from elfi.methods.bo.gpytorch_regression import GPyTorchRegression


class GPyTorchBayesianOptimization(BayesianOptimization):
    """Bayesian Optimization of an unknown target function."""

    def __init__(
        self,
        model,
        target_name=None,
        bounds=None,
        initial_evidence=None,
        update_interval=10,
        target_model=None,
        acquisition_method=None,
        acq_noise_var=0,
        exploration_rate=10,
        batch_size=1,
        batches_per_acquisition=None,
        async_acq=False,
        pyro_torch_prior=None,
        **kwargs
    ):
        """
        Initialize Bayesian optimization.

        :param model: ElfiModel, NodeReference
        :param target_name: str, NodeReference.+
            Only needed if model is an ElfiModel
        :param bounds : dict, optional
            The region where to estimate the posterior for each
            parameter in ``model.parameters``:
            ``dict('parameter_name':(lower, upper), ... )``.
            Not used if custom `target_model` is given.
        :param initial_evidence: int, dict, optional
            Number of initial evidence or a precomputed batch dict
            containing parameter and discrepancy values. Default value
            depends on the dimensionality.
        :param update_interval: int, optional
            How often to update the GP hyperparameters of the
            `target_model`.
        :param target_model: GPyRegression, optional
        :param acquisition_method: AnalyticAcquisitionFunction, optional
            Method of acquiring evidence points. Defaults to LCBSC.
        :param acq_noise_var: float, dict, optional
            Variance(s) of the noise added in the default LCBSC
            acquisition method. If a dictionary, values should be float
            specifying the variance for each dimension.
        :param exploration_rate: float, optional
            Exploration rate of the acquisition method
        :param batch_size: int, optional
            Elfi batch size. Defaults to 1.
        :param batches_per_acquisition: int, optional
            How many batches will be requested from the acquisition
            function at one go. Defaults to `max_parallel_batches`.
        :param async_acq: bool, optional
            Allow acquisitions to be made asynchronously, i.e., do not
            wait for all the results from the previous acquisition
            before making the next. This can be more efficient with a
            large amount of workers (e.g. in cluster environments) but
            foregoes the guarantee for the exactly same result with the
            same initial conditions (e.g. the seed). Default: False.
        :param pyro_torch_prior: torch.distributions, optional
            (Temporarily) needed as a by-pass of the internal processing
            of the elfi.Prior, if the trained model has to be sampled.
            Provide as the mathematically same object as the elfi.Prior.
            If a series of elfi.Priors was given for a multivariate
            distribution, give this as a multivariate distribution.
        :param **kwargs:
        """
        self.pyro_torch_prior = pyro_torch_prior
        target_model = target_model or GPyTorchRegression(
            (
                model.model.parameter_names
                if isinstance(model, NodeReference) else
                model.parameter_names
            ),
            bounds=bounds,
            initial_evidence=initial_evidence,
            **kwargs
        )
        super(GPyTorchBayesianOptimization, self).__init__(
            model,
            target_name=target_name,
            bounds=bounds,
            initial_evidence=initial_evidence,
            update_interval=update_interval,
            target_model=target_model,
            acquisition_method=acquisition_method,
            acq_noise_var=acq_noise_var,
            exploration_rate=exploration_rate,
            batch_size=batch_size,
            batches_per_acquisition=batches_per_acquisition,
            async_acq=async_acq,
            # **kwargs
        )
        if acquisition_method is None:
            self.acquisition_method = BoTorchLCBSC(
                self.target_model,
                noise_var=acq_noise_var,
                exploration_rate=exploration_rate,
                seed=self.seed
            )

    def extract_result(self):
        """
        Extract the result from the current state.
        Overrides the SciPy-based method in ELFI.

        :returns:
            OptimizationResult
        """
        # SciPy version for GPy.
        # x_min, _ = stochastic_optimization(
        #     self.target_model.predict_mean,
        #     self.target_model.bounds, seed=self.seed
        # )
        bounds = self.target_model.torch_bounds
        # Re-use the Upper Confidence Bound as Mean with beta = 0.
        # Reminder: ucb = mean + sqrt(beta * variance). With
        # maximize = False, it is ucb = -mean + sqrt(beta * variance).
        mean = UpperConfidenceBound(
            self.target_model._gp, beta=0, maximize=False
        )
        x_init = gen_batch_initial_conditions(
            mean,
            bounds,
            q=1,
            num_restarts=25,
            raw_samples=500 * 2**self.target_model.input_dim
        )
        # gen_candidates_scipy gives the minimum without noise.
        x_min, _ = gen_candidates_scipy(
            initial_conditions=x_init,
            acquisition_function=mean,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
        )
        x_min = x_min[0].T.detach().numpy()

        batch_min = arr2d_to_batch(
            x_min, self.target_model.parameter_names
        )
        outputs = arr2d_to_batch(
            self.target_model.X.detach().numpy(),
            self.target_model.parameter_names
        )

        # batch_min = arr2d_to_batch(x_min, self.parameter_names)
        # outputs = arr2d_to_batch(
        #     self.target_model.X, self.parameter_names
        # )
        outputs[self.target_name] = self.target_model.Y[0]

        return OptimizationResult(
            x_min=batch_min,
            outputs=outputs,
            **self._extract_result_kwargs()
        )


# The order of inheritance is the order as it is written here.
# Hence, GPyTorchBayesianOptimization.__init__ gets inherited,
# not BOLFI.__init__, which was inherited from BayesianOptimization.
class GPyTorchBOLFI(GPyTorchBayesianOptimization, BOLFI):
    """
    Bayesian Optimization for Likelihood-Free Inference (BOLFI).

    Approximates the discrepancy function by a stochastic regression
    model. Discrepancy model is fit by sampling the discrepancy function
    at points decided by the acquisition function.
    The method implements the framework introduced in
    Gutmann & Corander, 2016.

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood
    -Free Inference of Simulator-Based Statistical Models.
    JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html
    """

    def extract_posterior(self, threshold=None):
        """
        Return an object representing the approximate posterior.

        The approximation is based on surrogate model regression.

        :param threshold: float, optional
            Discrepancy threshold for creating the posterior (log with
            log discrepancy).

        :returns:
            elfi.methods.posteriors.BolfiPosterior
        """
        if self.state['n_evidence'] == 0:
            raise ValueError(
                'Model is not fitted yet, please see the `fit` method.')

        # Original ELFI implementation.
        # prior = ModelPrior(
        #     self.model,
        #     parameter_names=self.target_model.parameter_names
        # )
        # if self.pyro_torch_prior is None:
        #     print(
        #         "Warning: Translation from elfi.model.extensions."
        #         "ModelPrior into a PyTorch-compatible distribution "
        #         "not implemented."
        #     )
        # prior = self.pyro_torch_prior
        prior = ModelPrior(
            self.model,
            parameter_names=self.target_model.parameter_names
        )
        if threshold is None:
            bolfi_model = self.target_model._gp
            bounds = self.target_model.torch_bounds
            mean = UpperConfidenceBound(
                bolfi_model, beta=0, maximize=False
            )
            x_init = gen_batch_initial_conditions(
                mean,
                bounds,
                q=1,
                num_restarts=25,
                raw_samples=500 * 2**self.target_model.input_dim
            )
            # gen_candidates_scipy gives the minimum without noise.
            batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=x_init,
                acquisition_function=mean,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
            candidate = get_best_candidates(
                batch_candidates, -batch_acq_values
            )
            with gpytorch.settings.fast_computations(False, False, False):
                threshold = bolfi_model.likelihood(
                    bolfi_model(candidate)
                ).mean.cpu().detach().numpy()
        return BolfiPosterior(
            self.target_model, threshold=threshold, prior=prior
        )

    def sample(
        self,
        n_samples,
        warmup=None,
        n_chains=4,
        threshold=None,
        initials=None,
        algorithm='nuts',
        n_evidence=None,
        **kwargs
    ):
        r"""
        Sample the posterior distribution of BOLFI.

        Here the likelihood is defined through the cumulative density
        function of the standard normal distribution:
        L(\theta) \propto F((h-\mu(\theta)) / \sigma(\theta))
        where h is the threshold, and \mu(\theta) and \sigma(\theta) are
        the posterior mean and (noisy) standard deviation of the
        associated Gaussian process. The sampling is performed with an
        MCMC sampler (the No-U-Turn Sampler, NUTS).

        :param n_samples: int
            Number of requested samples from the posterior for each
            chain. This includes warmup, and note that the effective
            sample size is usually considerably smaller.
        :param warmup: int, optional
            Length of warmup sequence in MCMC sampling. Defaults to
            ``n_samples//2``.
        :param n_chains: int, optional
            Number of independent chains.
        :param threshold: float, optional
            The threshold (bandwidth) for posterior (give as log if log
            discrepancy).
        :param initials: array of shape (n_chains, n_params), optional
            Initial values for the sampled parameters for each chain.
            Defaults to best evidence points.
        :param algorithm: string, optional
            Sampling algorithm to use. Currently 'nuts' (default) is
            supported only.
        :param n_evidence: int
            If the regression model is not fitted yet, specify the
            amount of evidence.
        :returns:
            BolfiSample
        """
        # ToDo: change from ELFI-NUTS to Pyro-NUTS.
        # The problem: Pyro-NUTS does not support unnormalized
        # distributions.

        if self.state['n_batches'] == 0:
            self.fit(n_evidence)

        # TODO: add more MCMC algorithms
        if algorithm not in ['nuts', 'metropolis']:
            raise ValueError("Unknown posterior sampler.")
        if algorithm == 'metropolis':
            raise NotImplementedError(
                "Metropolis sampler not implemented. Use 'nuts' insteaad."
            )

        posterior = self.extract_posterior(threshold)
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with smallest
        # discrepancy
        if initials is not None:
            if np.asarray(initials).shape != (
                n_chains, self.target_model.input_dim
            ):
                raise ValueError(
                    "The shape of initials must be (n_chains, n_params).")
        else:
            inds = torch.argsort(self.target_model.Y)
            initials = self.target_model.X[inds].detach().numpy()

        # enables caching for default RBF kernel
        self.target_model.is_sampling = True

        tasks_ids = []
        ii_initial = 0
        """
        if algorithm == 'metropolis':
            # ToDo: provide initial sigma proposals.
            sigma_proposals = resolve_sigmas(
                self.target_model.parameter_names,
                sigma_proposals,
                self.target_model.bounds
            )
        """

        # sampling is embarrassingly parallel, so depending on
        # self.client this may parallelize
        for ii in range(n_chains):
            seed = get_sub_seed(self.seed, ii)
            # discard bad initialization points
            while np.isinf(posterior.logpdf(initials[ii_initial])):
                ii_initial += 1
                if ii_initial == len(inds):
                    raise ValueError(
                        "BOLFI.sample: Cannot find enough acceptable "
                        "initialization points!"
                    )

            if algorithm == 'nuts':
                tasks_ids.append(
                    self.client.apply(
                        mcmc.nuts,
                        n_samples,
                        initials[ii_initial],
                        posterior.logpdf,
                        posterior.gradient_logpdf,
                        n_adapt=warmup,
                        seed=seed,
                        **kwargs))
            """
            elif algorithm == 'metropolis':
                tasks_ids.append(
                    self.client.apply(
                        mcmc.metropolis,
                        n_samples,
                        initials[ii_initial],
                        posterior.logpdf,
                        sigma_proposals,
                        warmup,
                        seed=seed,
                        **kwargs))
            """
            ii_initial += 1

        # get results from completed tasks or run sampling
        # (client-specific)
        chains = []
        for id in tasks_ids:
            chains.append(self.client.get_result(id))

        chains = np.asarray(chains)
        print(
            "{} chains of {} iterations acquired. Effective sample size and "
            "Rhat for each parameter:".format(n_chains, n_samples)
        )
        for ii, node in enumerate(self.target_model.parameter_names):
            print(node, mcmc.eff_sample_size(chains[:, :, ii]),
                  mcmc.gelman_rubin_statistic(chains[:, :, ii]))
        self.target_model.is_sampling = False

        return BolfiSample(
            method_name='BOLFI',
            chains=chains,
            parameter_names=self.target_model.parameter_names,
            warmup=warmup,
            threshold=float(posterior.threshold),
            n_sim=self.state['n_evidence'],
            seed=self.seed)

    def sample_torch_prototype(
        self,
        n_samples,
        warmup=None,
        n_chains=4,
        threshold=None,
        initials=None,
        algorithm='nuts',
        n_evidence=None,
        **kwargs
    ):
        r"""
        Sample the posterior distribution of BOLFI.

        Here the likelihood is defined through the cumulative density
        function of the standard normal distribution:
        L(\theta) \propto F((h-\mu(\theta)) / \sigma(\theta))
        where h is the threshold, and \mu(\theta) and \sigma(\theta) are
        the posterior mean and (noisy) standard deviation of the
        associated Gaussian process. The sampling is performed with an
        MCMC sampler (the No-U-Turn Sampler, NUTS).
        :param n_samples: int
            Number of requested samples from the posterior for each
            chain. This includes warmup, and note that the effective
            sample size is usually considerably smaller.
        :param warmup: int, optional
            Length of warmup sequence in MCMC sampling. Defaults to
            ``n_samples//2``.
        :param n_chains: int, optional
            Number of independent chains.
        :param threshold: float, optional
            The threshold (bandwidth) for posterior (give as log if log
            discrepancy).
        :param initials: array of shape (n_chains, n_params), optional
            Initial values for the sampled parameters for each chain.
            Defaults to best evidence points.
        :param algorithm: string, optional
            Sampling algorithm to use. Currently 'nuts' (default) is
            supported only.
        :param n_evidence: int
            If the regression model is not fitted yet, specify the
            amount of evidence.
        :returns:
            BolfiSample
        """

        if self.state['n_batches'] == 0:
            self.fit(n_evidence)

        if algorithm not in ['nuts',]:
            raise ValueError("Unknown posterior sampler.")

        posterior = self.extract_posterior(threshold)
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with smallest
        # discrepancy
        if initials is not None:
            if torch.asarray(initials).shape != (
                n_chains, self.target_model.input_dim
            ):
                raise ValueError(
                    "The shape of initials must be (n_chains, n_params)."
                )
        else:
            inds = torch.argsort(self.target_model.Y)
            initials = torch.asarray(self.target_model.X[inds])

        self.target_model.is_sampling = True

        print(
            "ToDo: check if CDF transform of prior distribution is the same "
            "as the distribution given by prior multiplied with CDF."
        )

        self.target_model._gp.eval()
        self.target_model._gp.likelihood.eval()

        def pyro_model():
            # Settings (matrices have "small" shape
            # sample_size x sample_size):
            # covar_root_decomposition False,
            # log_prob False,
            # fast_solves False
            x = self.pyro_torch_prior
            with gpytorch.settings.fast_computations(False, False, False):
                gp_eval = self.target_model._gp.likelihood(
                    self.target_model._gp(x)
                )
            # The transform from GP mean and variance to "GP CDF".
            t = pyro.distributions.transforms.CumulativeDistributionTransform(
                pyro.distributions.Normal(
                    threshold - gp_eval.mean, torch.sqrt(gp_eval.variance)
                )
            )
            # Uniform distribution that might lead to Likelihood
            # sampling.
            # lower_bounds = torch.tensor([
            #     interval[0] for interval in self.bounds.values()
            # ])
            # upper_bounds = torch.tensor([
            #     interval[1] for interval in self.bounds.values()
            # ])
            # u = torch.distributions.Uniform(
            #     lower_bounds, upper_bounds
            # )
            return pyro.sample(
                "posterior",
                pyro.distributions.TransformedDistribution(
                    posterior.prior, [t]
                )
            )

        # seed = get_sub_seed(self.seed, ii)
        # discard bad initialization points
        # while torch.isinf(torch.asarray(
        #     posterior.logpdf(initials[ii_initial])
        # )):
        #     ii_initial += 1
        #     if ii_initial == len(inds):
        #         raise ValueError(
        #             "BOLFI.sample: Cannot find enough acceptable "
        #             "initialization points!"
        #         )

        if algorithm == 'nuts':
            print("ToDo: pass initial parameters to Pyro-NUTS.")
            print("ToDo: pass seed to Pyro-NUTS.")
            print("ToDo: switch from Pyro to NumPyro for performance.")
            init_params, potential_fn, transforms, _ = (
                pyro.infer.mcmc.util.initialize_model(
                    pyro_model,
                    model_args=(initials[0],),
                    num_chains=n_chains,
                )
            )
            nuts_kernel = NUTS(potential_fn=potential_fn)
            mcmc_run = MCMC(
                nuts_kernel,
                num_samples=n_samples,
                warmup_steps=warmup,
                num_chains=n_chains,
                initial_params=init_params,
                transforms=transforms,
                disable_progbar=False,
            )
            mcmc_run.run()
            chains = mcmc_run.get_samples()  # ['posterior']
            print(chains)

        # get results from completed tasks or run sampling
        # (client-specific)
        chains = torch.asarray(chains)
        print(
            "{} chains of {} iterations acquired. Effective sample size and "
            "Rhat for each parameter:".format(n_chains, n_samples)
        )
        # for ii, node in enumerate(self.target_model.parameter_names):
        #     print(
        #         node,
        #         eff_sample_size(chains[:, :, ii]),
        #         gelman_rubin_statistic(chains[:, :, ii])
        #     )
        mcmc_run.summary(prob=0.95)
        self.target_model.is_sampling = False

        self.target_model._gp.train()
        self.target_model._gp.likelihood.train()

        return BolfiSample(
            method_name='BOLFI',
            chains=chains,
            parameter_names=self.target_model.parameter_names,
            warmup=warmup,
            threshold=float(posterior.threshold),
            n_sim=self.state['n_evidence'],
            seed=self.seed,
        )
