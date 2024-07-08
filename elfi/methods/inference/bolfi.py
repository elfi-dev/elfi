"""This module contains BayesianOptimization- and BOLFI-classes."""

__all__ = ['BayesianOptimization', 'BOLFI']

import logging

import matplotlib.pyplot as plt
import numpy as np

import elfi.methods.mcmc as mcmc
import elfi.visualization.interactive as visin
import elfi.visualization.visualization as vis
from elfi.loader import get_sub_seed
from elfi.methods.bo.acquisition import LCBSC
from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.utils import stochastic_optimization
from elfi.methods.inference.parameter_inference import ParameterInference
from elfi.methods.posteriors import BolfiPosterior
from elfi.methods.results import BolfiSample, OptimizationResult
from elfi.methods.utils import arr2d_to_batch, batch_to_arr2d, ceil_to_batch_size, resolve_sigmas
from elfi.model.extensions import ModelPrior

logger = logging.getLogger(__name__)


class BayesianOptimization(ParameterInference):
    """Bayesian Optimization of an unknown target function."""

    def __init__(self,
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
                 **kwargs):
        """Initialize Bayesian optimization.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        target_name : str or NodeReference
            Only needed if model is an ElfiModel
        bounds : dict, optional
            The region where to estimate the posterior for each parameter in
            model.parameters: dict('parameter_name':(lower, upper), ... )`. Not used if
            custom target_model is given.
        initial_evidence : int, dict, optional
            Number of initial evidence or a precomputed batch dict containing parameter
            and discrepancy values. Default value depends on the dimensionality.
        update_interval : int, optional
            How often to update the GP hyperparameters of the target_model
        target_model : GPyRegression, optional
        acquisition_method : Acquisition, optional
            Method of acquiring evidence points. Defaults to LCBSC.
        acq_noise_var : float or dict, optional
            Variance(s) of the noise added in the default LCBSC acquisition method.
            If a dictionary, values should be float specifying the variance for each dimension.
        exploration_rate : float, optional
            Exploration rate of the acquisition method
        batch_size : int, optional
            Elfi batch size. Defaults to 1.
        batches_per_acquisition : int, optional
            How many batches will be requested from the acquisition function at one go.
            Defaults to max_parallel_batches.
        async_acq : bool, optional
            Allow acquisitions to be made asynchronously, i.e. do not wait for all the
            results from the previous acquisition before making the next. This can be more
            efficient with a large amount of workers (e.g. in cluster environments) but
            forgoes the guarantee for the exactly same result with the same initial
            conditions (e.g. the seed). Default False.
        **kwargs

        """
        model, target_name = self._resolve_model(model, target_name)
        output_names = [target_name] + model.parameter_names
        super(BayesianOptimization, self).__init__(
            model, output_names, batch_size=batch_size, **kwargs)

        target_model = target_model or GPyRegression(
            self.model.parameter_names, bounds=bounds)

        self.target_name = target_name
        self.target_model = target_model

        n_precomputed = 0
        n_initial, precomputed = self._resolve_initial_evidence(
            initial_evidence)
        if precomputed is not None:
            params = batch_to_arr2d(precomputed, self.target_model.parameter_names)
            n_precomputed = len(params)
            self.target_model.update(params, precomputed[target_name])

        self.batches_per_acquisition = batches_per_acquisition or self.max_parallel_batches

        prior = ModelPrior(self.model, parameter_names=self.target_model.parameter_names)
        self.acquisition_method = acquisition_method or LCBSC(self.target_model,
                                                              prior=prior,
                                                              noise_var=acq_noise_var,
                                                              exploration_rate=exploration_rate,
                                                              seed=self.seed)

        self.n_initial_evidence = n_initial
        self.n_precomputed_evidence = n_precomputed
        self.update_interval = update_interval
        self.async_acq = async_acq

        self.state['n_evidence'] = self.n_precomputed_evidence
        self.state['last_GP_update'] = self.n_initial_evidence
        self.state['acquisition'] = []

    def _resolve_initial_evidence(self, initial_evidence):
        # Some sensibility limit for starting GP regression
        precomputed = None
        n_required = max(10, 2**self.target_model.input_dim + 1)
        n_required = ceil_to_batch_size(n_required, self.batch_size)

        if initial_evidence is None:
            n_initial_evidence = n_required
        elif np.isscalar(initial_evidence):
            n_initial_evidence = int(initial_evidence)
        else:
            precomputed = initial_evidence
            n_initial_evidence = len(precomputed[self.target_name])

        if n_initial_evidence < 0:
            raise ValueError('Number of initial evidence must be positive or zero '
                             '(was {})'.format(initial_evidence))
        elif n_initial_evidence < n_required:
            logger.warning('We recommend having at least {} initialization points for '
                           'the initialization (now {})'.format(n_required, n_initial_evidence))

        if precomputed is None and (n_initial_evidence % self.batch_size != 0):
            logger.warning('Number of initial_evidence %d is not divisible by '
                           'batch_size %d. Rounding it up...' % (n_initial_evidence,
                                                                 self.batch_size))
            n_initial_evidence = ceil_to_batch_size(
                n_initial_evidence, self.batch_size)

        return n_initial_evidence, precomputed

    @property
    def n_evidence(self):
        """Return the number of acquired evidence points."""
        return self.state.get('n_evidence', 0)

    @property
    def acq_batch_size(self):
        """Return the total number of acquisition per iteration."""
        return self.batch_size * self.batches_per_acquisition

    def set_objective(self, n_evidence=None):
        """Set objective for inference.

        You can continue BO by giving a larger n_evidence.

        Parameters
        ----------
        n_evidence : int
            Number of total evidence for the GP fitting. This includes any initial
            evidence.

        """
        if n_evidence is None:
            n_evidence = self.objective.get('n_evidence', self.n_evidence)

        if n_evidence < self.n_evidence:
            logger.warning(
                'Requesting less evidence than there already exists')

        self.objective['n_evidence'] = n_evidence
        self.objective['n_sim'] = n_evidence - self.n_precomputed_evidence

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        OptimizationResult

        """
        x_min, _ = stochastic_optimization(
            self.target_model.predict_mean, self.target_model.bounds, seed=self.seed)

        batch_min = arr2d_to_batch(x_min, self.target_model.parameter_names)
        outputs = arr2d_to_batch(self.target_model.X, self.target_model.parameter_names)

        # batch_min = arr2d_to_batch(x_min, self.parameter_names)
        # outputs = arr2d_to_batch(self.target_model.X, self.parameter_names)
        outputs[self.target_name] = self.target_model.Y

        return OptimizationResult(
            x_min=batch_min, outputs=outputs, **self._extract_result_kwargs())

    def update(self, batch, batch_index):
        """Update the GP regression model of the target node with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(BayesianOptimization, self).update(batch, batch_index)
        self.state['n_evidence'] += self.batch_size

        params = batch_to_arr2d(batch, self.target_model.parameter_names)
        self._report_batch(batch_index, params, batch[self.target_name])

        optimize = self._should_optimize()
        self.target_model.update(params, batch[self.target_name], optimize)
        if optimize:
            self.state['last_GP_update'] = self.target_model.n_evidence

    def prepare_new_batch(self, batch_index):
        """Prepare values for a new batch.

        Parameters
        ----------
        batch_index : int
            next batch_index to be submitted

        Returns
        -------
        batch : dict or None
            Keys should match to node names in the model. These values will override any
            default values or operations in those nodes.

        """
        t = self._get_acquisition_index(batch_index)

        # Check if we still should take initial points from the prior
        if t < 0:
            return

        # Take the next batch from the acquisition_batch
        acquisition = self.state['acquisition']
        if len(acquisition) == 0:
            acquisition = self.acquisition_method.acquire(
                self.acq_batch_size, t=t)

        batch = arr2d_to_batch(
            acquisition[:self.batch_size], self.target_model.parameter_names)
        self.state['acquisition'] = acquisition[self.batch_size:]

        return batch

    def _get_acquisition_index(self, batch_index):
        acq_batch_size = self.batch_size * self.batches_per_acquisition
        initial_offset = self.n_initial_evidence - self.n_precomputed_evidence
        starting_sim_index = self.batch_size * batch_index

        t = (starting_sim_index - initial_offset) // acq_batch_size
        return t

    # TODO: use state dict
    @property
    def _n_submitted_evidence(self):
        return self.batches.total * self.batch_size

    def _allow_submit(self, batch_index):
        if not super(BayesianOptimization, self)._allow_submit(batch_index):
            return False

        if self.async_acq:
            return True

        # Allow submitting freely as long we are still submitting initial evidence
        t = self._get_acquisition_index(batch_index)
        if t < 0:
            return True

        # Do not allow acquisition until previous acquisitions are ready (as well
        # as all initial acquisitions)
        acquisitions_left = len(self.state['acquisition'])
        if acquisitions_left == 0 and self.batches.has_pending:
            return False

        return True

    def _should_optimize(self):
        current = self.target_model.n_evidence + self.batch_size
        next_update = self.state['last_GP_update'] + self.update_interval
        return current >= self.n_initial_evidence and current >= next_update

    def _report_batch(self, batch_index, params, distances):
        str = "Received batch {}:\n".format(batch_index)
        fill = 6 * ' '
        for i in range(self.batch_size):
            str += "{}{} at {}\n".format(fill, distances[i].item(), params[i])
        logger.debug(str)

    def plot_state(self, **options):
        """Plot the GP surface.

        This feature is still experimental and currently supports only 2D cases.
        """
        f = plt.gcf()
        if len(f.axes) < 2:
            f, _ = plt.subplots(1, 2, figsize=(
                13, 6), sharex='row', sharey='row')

        gp = self.target_model

        # Draw the GP surface
        visin.draw_contour(
            gp.predict_mean,
            gp.bounds,
            self.target_model.parameter_names,
            title='GP target surface',
            points=gp.X,
            axes=f.axes[0],
            **options)

        # Draw the latest acquisitions
        if options.get('interactive'):
            point = gp.X[-1, :]
            if len(gp.X) > 1:
                f.axes[1].scatter(*point, color='red')

        displays = [gp.instance]

        if options.get('interactive'):
            from IPython import display
            displays.insert(
                0,
                display.HTML('<span><b>Iteration {}:</b> Acquired {} at {}</span>'.format(
                    len(gp.Y), gp.Y[-1][0], point)))

        # Update
        visin._update_interactive(displays, options)

        acq_index = self._get_acquisition_index(self.state['n_batches'])

        def acq(x):
            return self.acquisition_method.evaluate(x, acq_index)

        # Draw the acquisition surface
        visin.draw_contour(
            acq,
            gp.bounds,
            self.target_model.parameter_names,
            title='Acquisition surface',
            points=None,
            axes=f.axes[1],
            **options)

        if options.get('close'):
            plt.close()

    def plot_discrepancy(self, axes=None, **kwargs):
        """Plot acquired parameters vs. resulting discrepancy.

        Parameters
        ----------
        axes : plt.Axes or arraylike of plt.Axes

        Return
        ------
        axes : np.array of plt.Axes

        """
        return vis.plot_discrepancy(self.target_model,
                                    self.target_model.parameter_names,
                                    axes=axes,
                                    **kwargs)

    def plot_gp(self, axes=None, resol=50, const=None, bounds=None, true_params=None, **kwargs):
        """Plot pairwise relationships as a matrix with parameters vs. discrepancy.

        Parameters
        ----------
        axes : matplotlib.axes.Axes, optional
        resol : int, optional
            Resolution of the plotted grid.
        const : np.array, optional
            Values for parameters in plots where held constant. Defaults to minimum evidence.
        bounds: list of tuples, optional
            List of tuples for axis boundaries.
        true_params : dict, optional
            Dictionary containing parameter names with corresponding true parameter values.

        Returns
        -------
        axes : np.array of plt.Axes

        """
        return vis.plot_gp(self.target_model, self.target_model.parameter_names, axes,
                           resol, const, bounds, true_params, **kwargs)


class BOLFI(BayesianOptimization):
    """Bayesian Optimization for Likelihood-Free Inference (BOLFI).

    Approximates the discrepancy function by a stochastic regression model.
    Discrepancy model is fit by sampling the discrepancy function at points decided by
    the acquisition function.

    The method implements the framework introduced in Gutmann & Corander, 2016.

    References
    ----------
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.
    http://jmlr.org/papers/v17/15-017.html

    """

    def fit(self, n_evidence, threshold=None, bar=True):
        """Fit the surrogate model.

        Generates a regression model for the discrepancy given the parameters.

        Currently only Gaussian processes are supported as surrogate models.

        Parameters
        ----------
        n_evidence : int, required
            Number of evidence for fitting
        threshold : float, optional
            Discrepancy threshold for creating the posterior (log with log discrepancy).
        bar : bool, optional
            Flag to remove (False) the progress bar from output.

        """
        logger.info("BOLFI: Fitting the surrogate model...")
        if n_evidence is None:
            raise ValueError(
                'You must specify the number of evidence (n_evidence) for the fitting')

        self.infer(n_evidence, bar=bar)
        return self.extract_posterior(threshold)

    def extract_posterior(self, threshold=None):
        """Return an object representing the approximate posterior.

        The approximation is based on surrogate model regression.

        Parameters
        ----------
        threshold: float, optional
            Discrepancy threshold for creating the posterior (log with log discrepancy).

        Returns
        -------
        posterior : elfi.methods.posteriors.BolfiPosterior

        """
        if self.state['n_evidence'] == 0:
            raise ValueError(
                'Model is not fitted yet, please see the `fit` method.')

        prior = ModelPrior(self.model, parameter_names=self.target_model.parameter_names)
        return BolfiPosterior(self.target_model, threshold=threshold, prior=prior)

    def sample(self,
               n_samples,
               warmup=None,
               n_chains=4,
               threshold=None,
               initials=None,
               algorithm='nuts',
               sigma_proposals=None,
               n_evidence=None,
               **kwargs):
        r"""Sample the posterior distribution of BOLFI.

        Here the likelihood is defined through the cumulative density function
        of the standard normal distribution:

        L(\theta) \propto F((h-\mu(\theta)) / \sigma(\theta))

        where h is the threshold, and \mu(\theta) and \sigma(\theta) are the posterior mean and
        (noisy) standard deviation of the associated Gaussian process.

        The sampling is performed with an MCMC sampler (the No-U-Turn Sampler, NUTS).

        Parameters
        ----------
        n_samples : int
            Number of requested samples from the posterior for each chain. This includes warmup,
            and note that the effective sample size is usually considerably smaller.
        warmpup : int, optional
            Length of warmup sequence in MCMC sampling. Defaults to n_samples//2.
        n_chains : int, optional
            Number of independent chains.
        threshold : float, optional
            The threshold (bandwidth) for posterior (give as log if log discrepancy).
        initials : np.array of shape (n_chains, n_params), optional
            Initial values for the sampled parameters for each chain.
            Defaults to best evidence points.
        algorithm : string, optional
            Sampling algorithm to use. Currently 'nuts'(default) and 'metropolis' are supported.
        sigma_proposals : dict, optional
            Standard deviations for Gaussian proposals of each parameter for Metropolis
            Markov Chain sampler. Defaults to 1/10 of surrogate model bound lengths.
        n_evidence : int
            If the regression model is not fitted yet, specify the amount of evidence

        Returns
        -------
        BolfiSample

        """
        if self.state['n_batches'] == 0:
            self.fit(n_evidence)

        # TODO: add more MCMC algorithms
        if algorithm not in ['nuts', 'metropolis']:
            raise ValueError("Unknown posterior sampler.")

        posterior = self.extract_posterior(threshold)
        warmup = warmup or n_samples // 2

        # Unless given, select the evidence points with smallest discrepancy
        if initials is not None:
            if np.asarray(initials).shape != (n_chains, self.target_model.input_dim):
                raise ValueError(
                    "The shape of initials must be (n_chains, n_params).")
        else:
            inds = np.argsort(self.target_model.Y[:, 0])
            initials = np.asarray(self.target_model.X[inds])

        self.target_model.is_sampling = True  # enables caching for default RBF kernel

        tasks_ids = []
        ii_initial = 0
        if algorithm == 'metropolis':
            sigma_proposals = resolve_sigmas(self.target_model.parameter_names,
                                             sigma_proposals,
                                             self.target_model.bounds)

        # sampling is embarrassingly parallel, so depending on self.client this may parallelize
        for ii in range(n_chains):
            seed = get_sub_seed(self.seed, ii)
            # discard bad initialization points
            while np.isinf(posterior.logpdf(initials[ii_initial])):
                ii_initial += 1
                if ii_initial == len(inds):
                    raise ValueError(
                        "BOLFI.sample: Cannot find enough acceptable initialization points!")

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

            ii_initial += 1

        # get results from completed tasks or run sampling (client-specific)
        chains = []
        for id in tasks_ids:
            chains.append(self.client.get_result(id))

        chains = np.asarray(chains)
        print(
            "{} chains of {} iterations acquired. Effective sample size and Rhat for each "
            "parameter:".format(n_chains, n_samples))
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
