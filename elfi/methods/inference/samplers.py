"""This module contains sampling based inference methods."""

__all__ = ['Rejection', 'SMC', 'AdaptiveDistanceSMC', 'AdaptiveThresholdSMC']

import logging
from math import ceil

import numpy as np

import elfi.visualization.interactive as visin
from elfi.loader import get_sub_seed
from elfi.methods.density_ratio_estimation import (DensityRatioEstimation,
                                                   calculate_densratio_basis_sigma)
from elfi.methods.inference.parameter_inference import ParameterInference
from elfi.methods.results import Sample, SmcSample
from elfi.methods.utils import (GMDistribution, arr2d_to_batch,
                                weighted_sample_quantile, weighted_var)
from elfi.model.elfi_model import AdaptiveDistance
from elfi.model.extensions import ModelPrior
from elfi.utils import is_array

logger = logging.getLogger(__name__)


class Sampler(ParameterInference):
    def sample(self, n_samples, *args, **kwargs):
        """Sample from the approximate posterior.

        See the other arguments from the `set_objective` method.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate from the (approximate) posterior
        *args
        **kwargs

        Returns
        -------
        result : Sample

        """
        bar = kwargs.pop('bar', True)
        self.bar = bar
        return self.infer(n_samples, *args, bar=bar, **kwargs)

    def _extract_result_kwargs(self):
        kwargs = super(Sampler, self)._extract_result_kwargs()
        for state_key in ['threshold', 'accept_rate']:
            if state_key in self.state:
                kwargs[state_key] = self.state[state_key]
        if hasattr(self, 'discrepancy_name'):
            kwargs['discrepancy_name'] = self.discrepancy_name
        return kwargs


class Rejection(Sampler):
    """Parallel ABC rejection sampler.

    For a description of the rejection sampler and a general introduction to ABC, see e.g.
    Lintusaari et al. 2016.

    References
    ----------
    Lintusaari J, Gutmann M U, Dutta R, Kaski S, Corander J (2016). Fundamentals and
    Recent Developments in Approximate Bayesian Computation. Systematic Biology.
    http://dx.doi.org/10.1093/sysbio/syw077.

    """

    def __init__(self, model, discrepancy_name=None, output_names=None, **kwargs):
        """Initialize the Rejection sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        kwargs:
            See ParameterInference

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        output_names = [discrepancy_name] + model.parameter_names + (output_names or [])
        self.adaptive = isinstance(model[discrepancy_name], AdaptiveDistance)
        if self.adaptive:
            model[discrepancy_name].init_adaptation_round()
            # Summaries are needed as adaptation data
            self.sums = [sumstat.name for sumstat in model[discrepancy_name].parents]
            for k in self.sums:
                if k not in output_names:
                    output_names.append(k)
        super(Rejection, self).__init__(model, output_names, **kwargs)

        self.discrepancy_name = discrepancy_name

    def set_objective(self, n_samples, threshold=None, quantile=None, n_sim=None):
        """Set objective for inference.

        Parameters
        ----------
        n_samples : int
            number of samples to generate
        threshold : float
            Acceptance threshold
        quantile : float
            In between (0,1). Define the threshold as the p-quantile of all the
            simulations. n_sim = n_samples/quantile.
        n_sim : int
            Total number of simulations. The threshold will be the n_samples-th smallest
            discrepancy among n_sim simulations.

        """
        if quantile is None and threshold is None and n_sim is None:
            quantile = .01
        self.state = dict(samples=None, threshold=np.Inf,
                          n_sim=0, accept_rate=1, n_batches=0)

        if quantile:
            n_sim = ceil(n_samples / quantile)

        # Set initial n_batches estimate
        if n_sim:
            n_batches = ceil(n_sim / self.batch_size)
        else:
            n_batches = self.max_parallel_batches

        self.objective = dict(n_samples=n_samples,
                              threshold=threshold, n_batches=n_batches)

        # Reset the inference
        self.batches.reset()

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(Rejection, self).update(batch, batch_index)
        if self.state['samples'] is None:
            # Lazy initialization of the outputs dict
            self._init_samples_lazy(batch)
        self._merge_batch(batch)
        self._update_state_meta()
        self._update_objective_n_batches()

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        result : Sample

        """
        if self.state['samples'] is None:
            raise ValueError('Nothing to extract')

        if self.adaptive:
            self._update_distances()

        # Take out the correct number of samples
        outputs = dict()
        for k, v in self.state['samples'].items():
            outputs[k] = v[:self.objective['n_samples']]

        return Sample(outputs=outputs, **self._extract_result_kwargs())

    def _init_samples_lazy(self, batch):
        """Initialize the outputs dict based on the received batch."""
        samples = {}
        e_noarr = "Node {} output must be in a numpy array of length {} (batch_size)."
        e_len = "Node {} output has array length {}. It should be equal to the batch size {}."

        for node in self.output_names:
            # Check the requested outputs
            if node not in batch:
                raise KeyError(
                    "Did not receive outputs for node {}".format(node))

            nbatch = batch[node]
            if not is_array(nbatch):
                raise ValueError(e_noarr.format(node, self.batch_size))
            elif len(nbatch) != self.batch_size:
                raise ValueError(e_len.format(
                    node, len(nbatch), self.batch_size))

            # Prepare samples
            shape = (self.objective['n_samples'] +
                     self.batch_size, ) + nbatch.shape[1:]
            dtype = nbatch.dtype

            if node == self.discrepancy_name:
                # Initialize the distances to inf
                samples[node] = np.ones(shape, dtype=dtype) * np.inf
            else:
                samples[node] = np.empty(shape, dtype=dtype)

        self.state['samples'] = samples

    def _merge_batch(self, batch):
        # TODO: add index vector so that you can recover the original order
        samples = self.state['samples']

        # Add current batch to adaptation data
        if self.adaptive:
            observed_sums = [batch[s] for s in self.sums]
            self.model[self.discrepancy_name].add_data(*observed_sums)

        # Check acceptance condition
        if self.objective.get('threshold') is None:
            accepted = slice(None, None)
            num_accepted = self.batch_size
        else:
            accepted = batch[self.discrepancy_name] <= self.objective.get('threshold')
            accepted = np.all(np.atleast_2d(np.transpose(accepted)), axis=0)
            num_accepted = np.sum(accepted)

        # Put the acquired samples to the end
        if num_accepted > 0:
            for node, v in samples.items():
                v[-num_accepted:] = batch[node][accepted]

        # Sort the smallest to the beginning
        # note: last (-1) distance measure is used when distance calculation is nested
        sort_distance = np.atleast_2d(np.transpose(samples[self.discrepancy_name]))[-1]
        sort_mask = np.argsort(sort_distance)
        for k, v in samples.items():
            v[:] = v[sort_mask]

    def _update_state_meta(self):
        """Update `n_sim`, `threshold`, and `accept_rate`."""
        o = self.objective
        s = self.state
        s['threshold'] = s['samples'][self.discrepancy_name][o['n_samples'] - 1]
        s['accept_rate'] = min(1, o['n_samples'] / s['n_sim'])

    def _update_objective_n_batches(self):
        # Only in the case that the threshold is used
        if self.objective.get('threshold') is None:
            return

        s = self.state
        t, n_samples = [self.objective.get(k)
                        for k in ('threshold', 'n_samples')]

        # noinspection PyTypeChecker

        if s['samples']:
            accepted = s['samples'][self.discrepancy_name] <= t
            n_acceptable = np.sum(np.all(np.atleast_2d(np.transpose(accepted)), axis=0))
        else:
            n_acceptable = 0

        if n_acceptable == 0:
            # No acceptable samples found yet, increase n_batches of objective by one in
            # order to keep simulating
            n_batches = self.objective['n_batches'] + 1
        else:
            accept_rate_t = n_acceptable / s['n_sim']
            # Add some margin to estimated n_batches. One could also use confidence
            # bounds here
            margin = .2 * self.batch_size * int(n_acceptable < n_samples)
            n_batches = (n_samples / accept_rate_t + margin) / self.batch_size
            n_batches = ceil(n_batches)

        self.objective['n_batches'] = n_batches
        logger.debug('Estimated objective n_batches=%d' %
                     self.objective['n_batches'])

    def _update_distances(self):

        # Update adaptive distance node
        self.model[self.discrepancy_name].update_distance()

        # Recalculate distances in current sample
        nums = self.objective['n_samples']
        data = {s: self.state['samples'][s][:nums] for s in self.sums}
        ds = self.model[self.discrepancy_name].generate(with_values=data)

        # Sort based on new distance measure
        sort_distance = np.atleast_2d(np.transpose(ds))[-1]
        sort_mask = np.argsort(sort_distance)

        # Update state
        self.state['samples'][self.discrepancy_name] = sort_distance
        for k in self.state['samples'].keys():
            if k != self.discrepancy_name:
                self.state['samples'][k][:nums] = self.state['samples'][k][sort_mask]

        self._update_state_meta()

    def plot_state(self, **options):
        """Plot the current state of the inference algorithm.

        This feature is still experimental and only supports 1d or 2d cases.
        """
        displays = []
        if options.get('interactive'):
            from IPython import display
            displays.append(
                display.HTML('<span>Threshold: {}</span>'.format(self.state['threshold'])))

        visin.plot_sample(
            self.state['samples'],
            nodes=self.parameter_names,
            n=self.objective['n_samples'],
            displays=displays,
            **options)


class SMC(Sampler):
    """Sequential Monte Carlo ABC sampler."""

    def __init__(self, model, discrepancy_name=None, output_names=None, **kwargs):
        """Initialize the SMC-ABC sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        kwargs:
            See ParameterInference

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        output_names = [discrepancy_name] + model.parameter_names + (output_names or [])

        super(SMC, self).__init__(model, output_names, **kwargs)

        self._prior = ModelPrior(self.model)
        self.discrepancy_name = discrepancy_name
        self.state['round'] = 0
        self._populations = []
        self._rejection = None
        self._round_random_state = None
        self._quantiles = None

    def set_objective(self, n_samples, thresholds=None, quantiles=None):
        """Set objective for ABC-SMC inference.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        thresholds : list, optional
            List of thresholds for ABC-SMC
        quantiles : list, optional
            List of selection quantiles used to determine sample thresholds

        """
        if thresholds is None and quantiles is None:
            raise ValueError("Either thresholds or quantiles is required to run ABC-SMC.")

        if thresholds is None:
            rounds = len(quantiles) - 1
        else:
            rounds = len(thresholds) - 1

        # Take previous iterations into account in case continued estimation
        self.state['round'] = len(self._populations)
        rounds = rounds + self.state['round']

        if thresholds is None:
            thresholds = np.full((rounds+1), None)
            self._quantiles = np.concatenate((np.full((self.state['round']), None), quantiles))
        else:
            thresholds = np.concatenate((np.full((self.state['round']), None), thresholds))

        self.objective.update(
            dict(
                n_samples=n_samples,
                n_batches=self.max_parallel_batches,
                round=rounds,
                thresholds=thresholds))
        self._init_new_round()
        self._update_objective()

    def extract_result(self):
        """Extract the result from the current state.

        Returns
        -------
        SmcSample

        """
        # Extract information from the population
        pop = self._extract_population()
        self._populations.append(pop)
        return SmcSample(
            outputs=pop.outputs,
            populations=self._populations.copy(),
            weights=pop.weights,
            threshold=pop.threshold,
            **self._extract_result_kwargs())

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(SMC, self).update(batch, batch_index)
        self._rejection.update(batch, batch_index)

        if self._rejection.finished:
            self.batches.cancel_pending()
            if self.bar:
                self.progress_bar.update_progressbar(self.progress_bar.scaling + 1,
                                                     self.progress_bar.scaling + 1)
            if self.state['round'] < self.objective['round']:
                self._populations.append(self._extract_population())
                self.state['round'] += 1
                self._init_new_round()
        self._update_objective()

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
        if self.state['round'] == 0:
            # Use the actual prior
            return

        # Sample from the proposal, condition on actual prior
        params = GMDistribution.rvs(*self._gm_params, size=self.batch_size,
                                    prior_logpdf=self._prior.logpdf,
                                    random_state=self._round_random_state)

        batch = arr2d_to_batch(params, self.parameter_names)
        return batch

    def _init_new_round(self):

        self._set_rejection_round(self.state['round'])

        if self.state['round'] == 0 and self._quantiles is not None:
            self._rejection.set_objective(
                self.objective['n_samples'], quantile=self._quantiles[0])
        else:
            if self._quantiles is not None:
                self._set_threshold()
            self._rejection.set_objective(
                self.objective['n_samples'], threshold=self.current_population_threshold)

    def _set_rejection_round(self, round):

        self._update_round_info(self.state['round'])

        # Get a subseed for this round for ensuring consistent results for the round
        seed = self.seed if round == 0 else get_sub_seed(self.seed, round)
        self._round_random_state = np.random.RandomState(seed)
        self._rejection = Rejection(
            self.model,
            discrepancy_name=self.discrepancy_name,
            output_names=self.output_names,
            batch_size=self.batch_size,
            seed=seed,
            max_parallel_batches=self.max_parallel_batches)

    def _update_round_info(self, round):
        if self.bar:
            reinit_msg = 'ABC-SMC Round {0} / {1}'.format(
                round + 1, self.objective['round'] + 1)
            self.progress_bar.reinit_progressbar(
                scaling=(self.state['n_batches']), reinit_msg=reinit_msg)
        dashes = '-' * 16
        logger.info('%s Starting round %d %s' % (dashes, round, dashes))

    def _extract_population(self):
        sample = self._rejection.extract_result()
        # Append the sample object
        sample.method_name = "Rejection within SMC-ABC"
        means, w, cov = self._compute_weights_means_and_cov(sample)
        sample.means = means
        sample.weights = w
        sample.meta['cov'] = cov
        return sample

    def _compute_weights_means_and_cov(self, pop):
        params = np.column_stack(tuple([pop.outputs[p] for p in self.parameter_names]))

        if self._populations:
            q_logpdf = GMDistribution.logpdf(params, *self._gm_params)
            p_logpdf = self._prior.logpdf(params)
            w = np.exp(p_logpdf - q_logpdf)
        else:
            w = np.ones(pop.n_samples)

        means = params.copy()

        if np.count_nonzero(w) == 0:
            raise RuntimeError("All sample weights are zero. If you are using a prior "
                               "with a bounded support, this may be caused by specifying "
                               "a too small sample size.")

        # New covariance
        cov = 2 * np.diag(weighted_var(params, w))

        if not np.all(np.isfinite(cov)):
            logger.warning("Could not estimate the sample covariance. This is often "
                           "caused by majority of the sample weights becoming zero."
                           "Falling back to using unit covariance.")
            cov = np.diag(np.ones(params.shape[1]))

        return means, w, cov

    def _update_objective(self):
        """Update the objective n_batches."""
        n_batches = sum([pop.n_batches for pop in self._populations])
        self.objective['n_batches'] = n_batches + \
            self._rejection.objective['n_batches']

    def _set_threshold(self):
        previous_population = self._populations[self.state['round']-1]
        threshold = weighted_sample_quantile(
            x=previous_population.discrepancies,
            alpha=self._quantiles[self.state['round']],
            weights=previous_population.weights)
        logger.info('ABC-SMC: Selected threshold for next population %.3f' % (threshold))
        self.objective['thresholds'][self.state['round']] = threshold

    @property
    def _gm_params(self):
        sample = self._populations[-1]
        return sample.means, sample.cov, sample.weights

    @property
    def current_population_threshold(self):
        """Return the threshold for current population."""
        return self.objective['thresholds'][self.state['round']]


class AdaptiveDistanceSMC(SMC):
    """SMC-ABC sampler with adaptive threshold and distance function.

    Notes
    -----
    Algorithm 5 in Prangle (2017)

    References
    ----------
    Prangle D (2017). Adapting the ABC Distance Function. Bayesian
    Analysis 12(1):289-309, 2017.
    https://projecteuclid.org/euclid.ba/1460641065

    """

    def __init__(self, model, discrepancy_name=None, output_names=None, **kwargs):
        """Initialize the adaptive distance SMC-ABC sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        kwargs:
            See ParameterInference

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        if not isinstance(model[discrepancy_name], AdaptiveDistance):
            raise TypeError('This method requires an adaptive distance node.')

        # Initialise adaptive distance node
        model[discrepancy_name].init_state()
        # Add summaries in additional outputs as these are needed to update the distance node
        sums = [sumstat.name for sumstat in model[discrepancy_name].parents]
        if output_names is None:
            output_names = sums
        else:
            for k in sums:
                if k not in output_names:
                    output_names.append(k)
        super(AdaptiveDistanceSMC, self).__init__(model, discrepancy_name,
                                                  output_names=output_names, **kwargs)

    def set_objective(self, n_samples, rounds, quantile=0.5):
        """Set objective for adaptive distance ABC-SMC inference.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        rounds : int, optional
            Number of populations to sample
        quantile : float, optional
            Selection quantile used to determine sample thresholds

        """
        super(AdaptiveDistanceSMC, self).set_objective(ceil(n_samples/quantile),
                                                       quantiles=[1]*rounds)
        self.population_size = n_samples
        self.quantile = quantile

    def _extract_population(self):
        # Extract population and metadata based on rejection sample
        rejection_sample = self._rejection.extract_result()
        outputs = dict()
        for k in self.output_names:
            outputs[k] = rejection_sample.outputs[k][:self.population_size]
        meta = rejection_sample.meta
        meta['adaptive_distance_w'] = self.model[self.discrepancy_name].state['w'][-1]
        meta['threshold'] = max(outputs[self.discrepancy_name])
        meta['accept_rate'] = self.population_size/meta['n_sim']
        method_name = "Rejection within adaptive distance SMC-ABC"
        sample = Sample(method_name, outputs, self.parameter_names, **meta)

        # Append the sample object
        means, w, cov = self._compute_weights_means_and_cov(sample)
        sample.means = means
        sample.weights = w
        sample.meta['cov'] = cov
        return sample

    def _extract_result_kwargs(self):
        kwargs = super(AdaptiveDistanceSMC, self)._extract_result_kwargs()
        kwargs['adaptive_distance_w'] = [pop.adaptive_distance_w for pop in self._populations]
        return kwargs

    def _set_threshold(self):
        round = self.state['round']
        self.objective['thresholds'][round] = self._populations[round-1].threshold

    @property
    def current_population_threshold(self):
        """Return the threshold for current population."""
        return [np.inf] + [pop.threshold for pop in self._populations]


class AdaptiveThresholdSMC(SMC):
    """ABC-SMC sampler with adaptive threshold selection.

    References
    ----------
    Simola U, Cisewski-Kehe J, Gutmann M U, Corander J (2021). Adaptive
    Approximate Bayesian Computation Tolerance Selection. Bayesian Analysis.
    https://doi.org/10.1214/20-BA1211

    """

    def __init__(self,
                 model,
                 discrepancy_name=None,
                 output_names=None,
                 initial_quantile=0.20,
                 q_threshold=0.99,
                 densratio_estimation=None,
                 **kwargs):
        """Initialize the adaptive threshold SMC-ABC sampler.

        Parameters
        ----------
        model : ElfiModel or NodeReference
        discrepancy_name : str, NodeReference, optional
            Only needed if model is an ElfiModel
        output_names : list, optional
            Additional outputs from the model to be included in the inference result, e.g.
            corresponding summaries to the acquired samples
        initial_quantile : float, optional
            Initial selection quantile for the first round of adaptive-ABC-SMC
        q_threshold : float, optional
            Termination criteratia for adaptive-ABC-SMC
        densratio_estimation : DensityRatioEstimation, optional
            Density ratio estimation object defining parameters for KLIEP
        kwargs:
            See ParameterInference

        """
        model, discrepancy_name = self._resolve_model(model, discrepancy_name)
        output_names = [discrepancy_name] + model.parameter_names + (output_names or [])

        super(SMC, self).__init__(model, output_names, **kwargs)

        self._prior = ModelPrior(self.model)
        self.discrepancy_name = discrepancy_name
        self.state['round'] = 0
        self._populations = []
        self._rejection = None
        self._round_random_state = None
        self.q_threshold = q_threshold
        self.initial_quantile = initial_quantile

        self.densratio = densratio_estimation or DensityRatioEstimation(n=100,
                                                                        epsilon=0.001,
                                                                        max_iter=200,
                                                                        abs_tol=0.01,
                                                                        fold=5,
                                                                        optimize=False)

    def set_objective(self,
                      n_samples,
                      max_iter=10):
        """Set objective for ABC-SMC inference.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        thresholds : list, optional
            List of thresholds for ABC-SMC
        max_iter : int, optional
            Maximum number of iterations

        """
        rounds = max_iter - 1

        # Take previous iterations into account in case continued estimation
        self.state['round'] = len(self._populations)
        rounds = rounds + self.state['round']

        # Initialise threshold selection and adaptive quantile
        thresholds = np.full((rounds+1), None)
        self._quantiles = np.full((rounds+1), None)
        self._quantiles[0] = self.initial_quantile

        self.objective.update(
            dict(
                n_samples=n_samples,
                n_batches=self.max_parallel_batches,
                round=rounds,
                thresholds=thresholds))
        self._init_new_round()
        self._update_objective()

    def update(self, batch, batch_index):
        """Update the inference state with a new batch.

        Parameters
        ----------
        batch : dict
            dict with `self.outputs` as keys and the corresponding outputs for the batch
            as values
        batch_index : int

        """
        super(SMC, self).update(batch, batch_index)
        self._rejection.update(batch, batch_index)

        if self._rejection.finished:
            self.batches.cancel_pending()

            if self.bar:
                self.progress_bar.update_progressbar(self.progress_bar.scaling + 1,
                                                     self.progress_bar.scaling + 1)

            self._new_population = self._extract_population()

            if self.state['round'] < self.objective['round']:

                self._set_adaptive_quantile()

                if self._quantiles[self.state['round']+1] < self.q_threshold:
                    self._populations.append(self._new_population)
                    self.state['round'] += 1
                    self._init_new_round()

        self._update_objective()

    def _set_adaptive_quantile(self):
        """Set adaptively the new threshold for current population."""
        logger.info("ABC-SMC: Adapting quantile threshold...")

        sample_data_current = self._resolve_sample(backwards_index=0)
        sample_data_previous = self._resolve_sample(backwards_index=-1)

        if self.densratio.optimize:
            sigma = list(10.0 ** np.arange(-1, 6))
        else:
            sigma = calculate_densratio_basis_sigma(sample_data_current['sigma_max'],
                                                    sample_data_previous['sigma_max'])

        self.densratio.fit(x=sample_data_current['samples'],
                           y=sample_data_previous['samples'],
                           weights_x=sample_data_current['weights'],
                           weights_y=sample_data_previous['weights'],
                           sigma=sigma)

        max_value = self.densratio.max_ratio()
        max_value = 1.0 if max_value < 1.0 else max_value
        self._quantiles[self.state['round']+1] = max(1 / max_value, 0.05)
        logger.info('ABC-SMC: Estimated maximum density ratio %.5f' % (1 / max_value))

    def _resolve_sample(self, backwards_index):
        """Get properties of the samples used in ratio estimation."""
        if self.state['round'] + backwards_index < 0:
            return self._densityratio_initial_sample()
        elif backwards_index == 0:
            sample = self._new_population
        else:
            sample = self._populations[backwards_index]

        weights = sample.weights
        samples = sample.samples_array
        sample_sigma = np.sqrt(np.diag(sample.cov))
        sigma_max = np.min(sample_sigma)
        sample_data = dict(samples=samples, weights=weights, sigma_max=sigma_max)

        return sample_data

    def _densityratio_initial_sample(self):
        n_samples = self._new_population.weights.shape[0]
        samples = self._prior.rvs(size=n_samples, random_state=self._round_random_state)
        weights = np.ones(n_samples)
        sample_cov = np.atleast_2d(np.cov(samples.reshape(n_samples, -1), rowvar=False))
        sigma_max = np.min(np.sqrt(np.diag(sample_cov)))
        return dict(samples=samples,
                    weights=weights,
                    sigma_max=sigma_max)
