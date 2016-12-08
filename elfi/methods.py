import logging
import numpy as np
import dask
import copy

from time import sleep
from distributed import Client
import scipy.stats as ss

import elfi
from elfi import Discrepancy, Transform
from elfi.async import wait
from elfi.distributions import Prior, SMC_Distribution
from elfi.posteriors import BolfiPosterior
from elfi.utils import stochastic_optimization, weighted_var
from elfi.bo.gpy_model import GPyModel
from elfi.bo.acquisition import LCBAcquisition, SecondDerivativeNoiseMixin, RbfAtPendingPointsMixin

logger = logging.getLogger(__name__)

"""Implementations of some ABC algorithms.

ABCMethod : Base class
Rejection : Rejection ABC (threshold or quantile-based)
BOLFI     : Bayesian optimization based ABC
"""


# TODO: allow passing the bare InferenceTask object instead of distance_node and
# parameter_node
class ABCMethod(object):
    """Base class for ABC methods.

    Parameters
    ----------
    distance_node : Discrepancy
        The discrepancy node in inference model.
    parameter_nodes : a list of Operations
        The nodes representing the targets of inference.
    batch_size : int, optional
        The number of samples in each parallel batch (may affect performance).
    store : various (optional)
        Storage object for logging data from inference process.
        Each method may have certain requirements for the store.
        See elfi.core.prepare_store interface.
    """
    def __init__(self, distance_node=None, parameter_nodes=None, batch_size=1000,
                 store=None):

        if not isinstance(distance_node, Discrepancy):
            raise TypeError("Distance node needs to inherit elfi.Discrepancy")
        if not all(map(lambda n: isinstance(n, Transform), parameter_nodes)):
            raise TypeError("Parameter nodes need to inherit elfi.Operation")

        self.distance_node = distance_node
        self.parameter_nodes = parameter_nodes
        self.n_params = len(parameter_nodes)
        self.batch_size = batch_size
        self.store = elfi.core.prepare_store(store)

    def sample(self, n_samples, *args, **kwargs):
        """Run the sampler.

        Subsequent calls will reuse existing data without rerunning the
        simulator until necessary.

        Parameters
        ----------
        n_samples : int
            Number of samples from the posterior

        Returns
        -------
        A dictionary with at least the following items:
        samples : list of np.arrays
            Samples from the posterior distribution of each parameter.
        """
        raise NotImplementedError("Subclass implements")

    def _acquire(self, n_sim, verbose=True):
        """Acquires n_sim distances and parameters

        Parameters
        ----------
        n_sim : int
            number of values to compute

        Returns
        -------
        distances : np.ndarray with shape (n_sim, 1)
            Distance values matching with 'parameters'
        parameters: list containing np.ndarrays of shapes (n_sim, ...)
            Contains parameter values for each parameter node in order
        """
        if verbose:
            logger.info("{}: Running with {} proposals.".format(self.__class__.__name__, n_sim))
        distances = self.distance_node.acquire(n_sim, batch_size=self.batch_size).compute()
        parameters = [p.acquire(n_sim, batch_size=self.batch_size).compute()
                      for p in self.parameter_nodes]

        return distances, parameters


# TODO: allow setting the cache on for the distances
# TODO: allow vector thresholds?
class Rejection(ABCMethod):
    """Rejection sampler.
    """

    def sample(self, n_samples, quantile=0.01, threshold=None):
        """Run the rejection sampler.

        In quantile mode, the simulator is run (n/quantile) times.

        In threshold mode, the simulator is run until n_samples can be returned.
        Not that a poorly-chosen threshold may result in a never-ending loop.

        Parameters
        ----------
        n_samples : int
            Number of samples from the posterior.
        quantile : float in range ]0, 1], optional
            The quantile for determining the acceptance threshold.
        threshold : float, optional
            The acceptance threshold.

        Returns
        -------
        A dictionary with items:
        samples : list of np.arrays
            Samples from the posterior distribution of each parameter.
        threshold : float
            The threshold value used in inference.
        n_sim : int
            Number of simulated data sets.
        """

        if quantile <= 0 or quantile > 1:
            raise ValueError("Quantile must be in range ]0, 1].")

        if threshold is None:
            # Quantile case
            n_sim = int(np.ceil(n_samples / quantile))
            distances, parameters = self._acquire(n_sim)
            sorted_inds = np.argsort(distances, axis=0)
            threshold = distances[sorted_inds[n_samples-1]].item()
            # Preserve the order
            accepted = np.zeros(len(distances), dtype=bool)
            accepted[sorted_inds[:n_samples]] = True
        else:
            # Threshold case
            n_sim = self._ceil_to_batch_size(n_samples)
            while True:
                distances, parameters = self._acquire(n_sim)
                accepted, n_accpt, accpt_r = self._compute_acceptance(distances, threshold)
                if n_accpt >= n_samples:
                    break
                # Add 5% more. In the future perhaps use some bootstrapped CI
                n_sim = self._ceil_to_batch_size((n_samples / accpt_r) * 1.05)

        posteriors = [p[accepted][:n_samples] for p in parameters]

        return {'samples': posteriors, 'threshold': threshold, 'n_sim': n_sim}

    def _compute_acceptance(self, distances, threshold):
        accepted = distances[:,0] < threshold
        n_accepted = np.sum(accepted)
        accept_ratio = n_accepted/len(distances)
        return accepted, n_accepted, accept_ratio

    def reject(self, threshold, n_sim=None):
        """Return samples below rejection threshold.

        Parameters
        ----------
        threshold : float
            The acceptance threshold.
        n_sim : int, optional
            Number of simulations from which to reject
            Defaults to the number of finished simulations.

        Returns
        -------
        A dictionary with items:
        samples : list of np.arrays
            Samples from the posterior distribution of each parameter.
        threshold : float
            The threshold value used in inference.
        n_sim : int
            Number of simulated data sets.
        """

        # TODO: add method to core
        if n_sim is None:
            n_sim = self.distance_node._generate_index

        distances, parameters = self._acquire(n_sim)
        distances = distances.ravel()  # avoid unnecessary indexing

        accepted = distances < threshold
        posteriors = [p[accepted] for p in parameters]

        return {'samples': posteriors, 'threshold': threshold, 'n_sim': n_sim}

    def _ceil_to_batch_size(self, n):
        return int(np.ceil(np.ceil(n) / self.batch_size) * self.batch_size)


class SMC(Rejection):
    """Likelihood-free sequential Monte Carlo sampler.

    Based on Algorithm 4 in:
    Jean-Michel Marin, Pierre Pudlo, Christian P Robert, and Robin J Ryder:
    Approximate bayesian computational methods, Statistics and Computing,
    22(6):1167–1180, 2012.

    Parameters
    ----------
    (as in Rejection)

    See Also
    --------
    `Rejection` : Basic rejection sampling.
    """

    def sample(self, n_samples, n_populations, schedule):
        """Run SMC-ABC sampler.

        Parameters
        ----------
        n_samples : int
            Number of samples drawn from the posterior.
        n_populations : int
            Number of particle populations to iterate over.
        schedule : iterable of floats in range ]0, 1]
            Acceptance quantiles for particle populations.

        Returns
        -------
        A dictionary with items:
        samples : list of np.arrays
            Samples from the posterior distribution of each parameter.
        samples_history : list of lists of np.arrays
            Samples from previous populations.
        weighted_sds_history : list of lists of floats
            Weighted standard deviations from previous populations.
        """

        # initialize with rejection sampling
        result = super(SMC, self).sample(n_samples, quantile=schedule[0])
        parameters = result['samples']
        weights = np.ones(n_samples)

        # save original priors using deepcopy to capture possible conditionality
        orig_priors = copy.deepcopy(self.parameter_nodes)

        params_history = []
        weighted_sds_history = []
        try:  # Try is used, since Priors will change. The `finally` part always reverts to original.
            for tt in range(1, n_populations):
                params_history.append(parameters)

                weights /= np.sum(weights)  # normalize weights here

                # calculate weighted standard deviations
                weighted_sds = [ np.sqrt( 2. * weighted_var(p, weights) )
                                 for p in parameters ]
                weighted_sds_history.append(weighted_sds)

                # set new prior distributions based on previous samples
                for ii, p in enumerate(self.parameter_nodes):
                    pname = "{}_smc{}".format(p.name, tt)
                    new_prior = Prior(pname, SMC_Distribution, parameters[ii],
                                      weighted_sds[ii], weights)
                    self.parameter_nodes[ii] = p.change_to(new_prior,
                                                           transfer_parents=False,
                                                           transfer_children=True)

                # rejection sampling with the new priors
                result = super(SMC, self).sample(n_samples, quantile=schedule[tt])
                parameters = result['samples']

                # calculate new unnormalized weights for parameters
                weights_new = np.ones(n_samples)
                for ii in range(self.n_params):
                    weights_denom = np.sum(weights *
                                           self.parameter_nodes[ii].pdf(parameters[ii]).T)

                    weights_new *= orig_priors[ii].pdf(parameters[ii], all_samples=parameters).ravel()
                    weights_new /= weights_denom
                weights = weights_new

        finally:
            # revert to original priors
            self.parameter_nodes = [p.change_to(orig_priors[ii],
                                                transfer_parents=False,
                                                transfer_children=True)
                                    for ii, p in enumerate(self.parameter_nodes)]

        return {'samples': parameters,
                'samples_history': params_history,
                'weighted_sds_history': weighted_sds_history}


class BolfiAcquisition(SecondDerivativeNoiseMixin, LCBAcquisition):
    """Default acquisition function for BOLFI.
    """
    pass


class AsyncBolfiAcquisition(SecondDerivativeNoiseMixin,
                            RbfAtPendingPointsMixin,
                            LCBAcquisition):
    """Default acquisition function for BOLFI (async case).
    """
    pass


class BOLFI(ABCMethod):
    """BOLFI ABC inference

    Approximates the true discrepancy function by a stochastic regression model.
    Model is fit by sampling the true discrepancy function at points decided by
    the acquisition function.

    Parameters
    ----------
    distance_node : Discrepancy
    parameter_nodes : a list of Operations
    batch_size : int, optional
    store : various (optional)
        Storage object that implements elfi.storage.NameIndexDataInterface
    model : stochastic regression model object (eg. GPyModel)
        Model to use for approximating the discrepancy function.
    acquisition : acquisition function object (eg. AcquisitionBase derivate)
        Policy for selecting the locations where discrepancy is computed
    sync : bool
        Whether to sample sychronously or asynchronously
    bounds : list of tuples (min, max) per dimension
        The region where to estimate the posterior (box-constraint)
    client : dask Client
        Client to use for computing the discrepancy values
    n_surrogate_samples : int
        Number of points to calculate discrepancy at if 'acquisition' is not given
    optimizer : string
        See GPyModel
    n_opt_iters : int
        See GPyModel
    """

    def __init__(self, distance_node=None, parameter_nodes=None, batch_size=10,
                 store=None, model=None, acquisition=None, sync=True,
                 bounds=None, client=None, n_surrogate_samples=10,
                 optimizer="scg", n_opt_iters=0):
        super(BOLFI, self).__init__(distance_node, parameter_nodes, batch_size, store)
        self.n_dimensions = len(self.parameter_nodes)
        self.model = model or GPyModel(self.n_dimensions, bounds=bounds,
                                       optimizer=optimizer, n_opt_iters=n_opt_iters)
        self.sync = sync
        if acquisition is not None:
            self.acquisition = acquisition
        elif sync is True:
            self.acquisition = BolfiAcquisition(self.model,
                                                n_samples=n_surrogate_samples)
        else:
            self.acquisition = AsyncBolfiAcquisition(self.model,
                                                     n_samples=n_surrogate_samples)
        if client is not None:
            self.client = client
        else:
            logger.debug("{}: No dask client given, creating a local client."
                    .format(self.__class__.__name__))
            self.client = Client()
            dask.set_options(get=self.client.get)

        if self.store is not None:
            if not isinstance(self.store, elfi.storage.NameIndexDataInterface):
                raise ValueError("Expected storage object to fulfill NameIndexDataInterface")
            self.sample_idx = 0
            self._log_model()

    def _log_model(self):
        if self.store is not None:
            # TODO: What should name be if we have multiple BOLFI inferences?
            self.store.set("BOLFI-model", self.sample_idx, [self.model.copy()])
            self.sample_idx += 1

    def infer(self, threshold=None):
        """Bolfi inference.

        Parameters
        ----------
        see get_posterior

        Returns
        -------
        see get_posterior
        """
        self.create_surrogate_likelihood()
        return self.get_posterior(threshold)

    def create_surrogate_likelihood(self):
        """Samples discrepancy iteratively to fit the surrogate model.
        """
        if self.sync is True:
            logger.info("{}: Sampling {:d} samples in batches of {:d}"
                    .format(self.__class__.__name__,
                            self.acquisition.samples_left,
                            self.batch_size))
        else:
            logger.info("{}: Sampling {:d} samples asynchronously {:d} samples in parallel"
                    .format(self.__class__.__name__,
                            self.acquisition.samples_left,
                            self.batch_size))
        futures = list()  # pending future results
        pending = list()  # pending locations matched to futures by list index
        while (not self.acquisition.finished) or (len(pending) > 0):
            next_batch_size = self._next_batch_size(len(pending))
            if next_batch_size > 0:
                pending_locations = np.atleast_2d(pending) if len(pending) > 0 else None
                new_locations = self.acquisition.acquire(next_batch_size, pending_locations)
                for location in new_locations:
                    wv_dict = {param.name: np.atleast_2d(location[i])
                               for i, param in enumerate(self.parameter_nodes)}
                    future = self.distance_node.generate(1, with_values=wv_dict)
                    futures.append(future)
                    pending.append(location)
            result, result_index, futures = wait(futures, self.client)
            location = pending.pop(result_index)
            logger.debug("{}: Observed {:f} at {}."
                    .format(self.__class__.__name__, result[0][0], location))
            self.model.update(location[None,:], result)
            self._log_model()

    def _next_batch_size(self, n_pending):
        """Returns batch size for acquisition function.
        """
        if self.sync is True and n_pending > 0:
            return 0
        return min(self.batch_size, self.acquisition.samples_left) - n_pending

    def get_posterior(self, threshold):
        """Returns the posterior.

        Parameters
        ----------
        threshold: float
            discrepancy threshold for creating the posterior

        Returns
        -------
        BolfiPosterior object
        """
        return BolfiPosterior(self.model, threshold)

