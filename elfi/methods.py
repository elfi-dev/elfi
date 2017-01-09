import logging
from functools import partial

import numpy as np
import dask

from elfi import core
from elfi import Discrepancy, Transform
from elfi import storage
from elfi.async import wait, next_result
from elfi.env import client as elfi_client
from elfi.distributions import Prior, SMCProposal
from elfi.posteriors import BolfiPosterior
from elfi.bo.gpy_model import GPyModel
from elfi.bo.acquisition import LCBAcquisition, SecondDerivativeNoiseMixin, RbfAtPendingPointsMixin

logger = logging.getLogger(__name__)

"""Implementations of some ABC algorithms.

ABCMethod : Base class
Rejection : Rejection ABC (threshold or quantile-based)
BOLFI     : Bayesian optimization based ABC
"""


# TODO: make a result class
# TODO: allow passing the bare InferenceTask object
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
        self.batch_size = int(batch_size)
        self.store = core.prepare_store(store)

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

    def estimate_batches_needed(self, n_samples, accept_rate):
        """Gives an estimate for how many batches of size `self.batch_size` are needed
        to generate n_samples

        Parameters
        ----------
        n_samples : int
            The amount of samples
        accept_rate : float
            estimate of the accept rate

        Returns
        -------
        int
        """
        return self.estimate_proposals_needed(n_samples, accept_rate) // self.batch_size

    def estimate_proposals_needed(self, n_samples, accept_rate):
        """Estimate the number of proposals needed in order to generate
        n_samples of samples.

        Parameters
        ----------
        n_samples : int
            The amount of samples
        accept_rate : float

        Returns
        -------
        int
            The returned estimate is divisible with `self.batch_size`
        """

        with np.errstate(divide='ignore'):
            n = n_samples / accept_rate
        if np.isinf(n):
            n = self.ncores*self.batch_size

        return self._ceil_in_batch_sizes(n)

    def _ceil_in_batch_sizes(self, n):
        """Rounds up n to the nearest higher integer divisible with `self.batch_size`."""
        return int(np.ceil(n / self.batch_size)) * self.batch_size

    def _acquire(self, n_sim, verbose=True):
        """Acquires n_sim distances and parameters

        Parameters
        ----------
        n_sim : int
            number of values to compute

        Returns
        -------
        distances : np.ndarray
        parameters: list
            containing np.ndarray objects of shapes (n_sim, ...) that contain the
            parameter values for their respective parameter nodes
        """
        if verbose:
            logger.info("{}: Running with {} proposals.".format(self.__class__.__name__,
                                                                n_sim))
        distances = self.distance_node.acquire(n_sim,
                                               batch_size=self.batch_size).compute()
        parameters = [p.acquire(n_sim, batch_size=self.batch_size).compute()
                      for p in self.parameter_nodes]

        return distances, parameters

    @property
    def ncores(self):
        """Total number of cores available in elfi.client."""
        return sum(elfi_client().ncores().values())

    @classmethod
    def compute_acceptance(cls, distances, threshold):
        accepted = cls.accepted(distances, threshold)
        n_accepted = np.sum(accepted)
        accept_rate = n_accepted/len(distances)
        return accepted, n_accepted, accept_rate

    @staticmethod
    def accepted(distances, threshold):
        """

        Returns
        -------
        np.array
        """
        return distances[:,0] <= threshold


# TODO: make asynchronous so that it can handle larger arrays than would fit in memory
# TODO: allow vector thresholds?
class Rejection(ABCMethod):
    """Rejection sampler.
    """

    def sample(self, n_samples, quantile=0.01, threshold=None):
        """Run the rejection sampler.

        In quantile mode, the simulator is run (n/quantile) times.

        In threshold mode, the simulator is run until n_samples can be returned.
        Note that a poorly-chosen threshold may result in a never-ending loop.

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

        parameters = None; distances = None; accepted = None; accept_rate = None

        if threshold is None:
            # Quantile case
            n_sim = int(np.ceil(n_samples / quantile))
            distances, parameters = self._acquire(n_sim)
            sorted_inds = np.argsort(distances, axis=0)
            threshold = distances[sorted_inds[n_samples-1]].item()
            # Preserve the order
            accepted = np.zeros(len(distances), dtype=bool)
            accepted[sorted_inds[:n_samples]] = True
            accept_rate = quantile
        else:
            # Threshold case
            n_sim = self._ceil_in_batch_sizes(n_samples)
            while True:
                distances, parameters = self._acquire(n_sim)
                accepted, n_accepted, accept_rate = self.compute_acceptance(distances,
                                                                            threshold)
                if n_accepted >= n_samples:
                    break

                n_sim = self.estimate_proposals_needed(n_samples, accept_rate)

        samples = [p[accepted][:n_samples] for p in parameters]
        distances = distances[accepted][:n_samples]

        return dict(samples=samples,
                    distances=distances,
                    threshold=threshold,
                    n_sim=n_sim,
                    accept_rate=accept_rate)

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
        accepted = distances[:, 0] < threshold
        samples = [p[accepted] for p in parameters]
        distances = distances[accepted]
        accept_rate = sum(accepted)/n_sim

        return dict(samples=samples,
                    distances=distances,
                    threshold=threshold,
                    n_sim=n_sim,
                    accept_rate=accept_rate)


# TODO: add tests
def smc_prior_transform(input_dict, column_interval, prior):
    if not isinstance(column_interval, tuple):
        column_interval = (column_interval, column_interval+1)
    idata = input_dict['data']
    data = idata[0][:, column_interval[0]:column_interval[1]]
    # Must call ravel, because pdf retains shape in non multidimensional distributions
    pdf = prior.pdf(data, *idata[1:]).ravel()
    return core.to_output_dict(input_dict, data=data, pdf=pdf)


class SMC(ABCMethod):
    """Likelihood-free sequential Monte Carlo sampler.

    Based on Algorithm 4 in:
    Jarno Lintusaari, Michael U. Gutmann, Ritabrata Dutta, Samuel Kaski, Jukka Corander (2016).
    Fundamentals and Recent Developments in Approximate Bayesian Computation.
    Systematic Biology, doi: 10.1093/sysbio/syw077.
    http://sysbio.oxfordjournals.org/content/early/2016/09/07/sysbio.syw077.full

    See Also
    --------
    `Rejection` : Basic rejection sampling.
    """

    def __init__(self, *args, **kwargs):
        self.distance_futures = []
        self.parameter_futures = []
        self.batch_indexes = []
        self._batches_count = 0
        super(SMC, self).__init__(*args, **kwargs)

    def sample(self, n_samples, n_populations, schedule):
        """Run SMC-ABC sampler.

        Parameters
        ----------
        n_samples : int
            Number of samples drawn from the posterior.
        n_populations : int
            Number of particle populations to iterate over.
        schedule : iterable of floats
            Thresholds for particle populations.

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

        # Run first round with standard rejection sampling
        logger.info("SMC initialization with Rejection sampling")
        rej = Rejection(self.distance_node, self.parameter_nodes, batch_size=self.batch_size)
        result = rej.sample(n_samples, threshold=schedule[0])
        samples = result['samples']
        distances = result['distances']
        accept_rate = result['accept_rate']
        threshold = result['threshold']
        n_sim = result['n_sim']
        weights = [1]*n_samples

        # Build the SMC proposal
        q = SMCProposal(np.hstack(samples), weights)
        qnode = Prior("smc_proposal", q,
                      size=(q.size),
                      inference_task=self.distance_node.inference_task)

        # Connect the proposal to the graph
        for i, p in enumerate(self.parameter_nodes):
            p.add_parent(qnode, index=0)
            # TODO: handle multivariate prior by investigating the result data
            transform = partial(smc_prior_transform,
                                column_interval=i, prior=p.distribution)
            p.set_transform(transform)

        # TODO: remove this once core allows starting from non zero index
        for node in qnode.component:
            node.reset(propagate=False)

        samples_history = []; distances_history = []; threshold_history = []
        weights_history = []; accept_rate_history = []; n_sim_history = []

        # Start iterations
        for t in range(1, n_populations):
            logger.info("SMC starting iteration {}".format(t))
            # Update the proposal
            if t > 1:
                q.set_population(np.hstack(samples), weights)

            samples_history.append([s.copy() for s in samples])
            distances_history.append(distances.copy())
            threshold_history.append(threshold)
            weights_history.append(weights.copy())
            accept_rate_history.append(accept_rate)
            n_sim_history.append(n_sim)

            threshold = schedule[t]
            n_accepts_in_sample = np.sum(self.accepted(distances_history[-1], threshold))
            # Heuristic estimate for the new accept rate
            accept_rate = np.mean([n_accepts_in_sample/n_sim_history[-1], accept_rate])
            n_accepts = 0
            n_sim = 0

            self._batches_count = 0
            samples_t = [[] for p in self.parameter_nodes]
            distances_t = []
            weights_t = []

            # Start adding batches until n_samples is achieved
            while True:
                n_add = self._add_new_batches(n_samples, n_accepts, accept_rate)
                # Fill the lists with Nones
                for l in samples_t:
                    l += [None]*n_add
                distances_t += [None]*n_add
                weights_t += [None]*n_add

                wait(self.distance_futures)

                # Iterate over all finished futures
                while True:
                    distances_batch, i_future = next_result(self.distance_futures)
                    if i_future is None:
                        break

                    p_futures_batch = self.parameter_futures.pop(i_future)
                    batch_index = self.batch_indexes.pop(i_future)

                    accepted_new = self.accepted(distances_batch, threshold)
                    n_new = np.sum(accepted_new)

                    new_prior_pdf = np.ones(n_new)
                    new_samples = np.zeros((n_new, q.size[0]))

                    for p_index, p_out in enumerate(p_futures_batch):
                        p_out = p_out.result()
                        new_samples[:, p_index:p_index+1] = p_out['data'][accepted_new]
                        new_prior_pdf *= p_out['pdf'][accepted_new]
                        samples_t[p_index][batch_index] = new_samples[:, p_index:p_index+1]

                    distances_t[batch_index] = distances_batch[accepted_new]
                    weights_t[batch_index] = new_prior_pdf / q.pdf(new_samples)

                    n_accepts += n_new
                    n_sim += len(accepted_new)
                    accept_rate = n_accepts / n_sim

                if n_samples <= n_accepts and len(self.distance_futures) == 0:
                    break

            for i_p, p in enumerate(samples):
                p[:] = np.vstack(samples_t[i_p])[:n_samples]
            distances[:] = np.vstack(distances_t)[:n_samples]
            # TODO: make weights 2d as well
            weights[:] = np.concatenate(weights_t)[:n_samples]

        return dict(samples=samples,
                    distances=distances,
                    weights=weights,
                    threshold=threshold,
                    n_sim=n_sim,
                    accept_rate=accept_rate,
                    samples_history=samples_history,
                    distances_history=distances_history,
                    weights_history=weights_history,
                    threshold_history=threshold_history,
                    accept_rate_history=accept_rate_history)

    def _add_new_batches(self, n_samples, n_accepted, accept_rate):
        n_left = n_samples - n_accepted
        n_batches = self.estimate_batches_needed(n_left, accept_rate)
        n_add = max(0, n_batches - len(self.distance_futures))

        for i_batch in range(n_add):
            logger.info("SMC generating a batch of {} ({}/{})".format(self.batch_size,
                                                                      i_batch+1,
                                                                      n_add))
            # TODO: self.distance_node.create_delayed_output(self.batch_size)
            d = self.distance_node.generate(self.batch_size)
            p = [pn.get_delayed_output(d) for pn in self.parameter_nodes]
            futures = elfi_client().compute([d]+p)
            self.distance_futures.append(futures[0])
            self.parameter_futures.append(futures[1:])
            self.batch_indexes.append(self._batches_count)
            self._batches_count += 1

        return n_add


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
    max_opt_iters : int
        See GPyModel
    """

    def __init__(self, distance_node=None, parameter_nodes=None, batch_size=10,
                 store=None, model=None, acquisition=None, sync=True,
                 bounds=None, client=None, n_surrogate_samples=10,
                 optimizer="scg", n_opt_iters=0):
        super(BOLFI, self).__init__(distance_node, parameter_nodes, batch_size, store)
        self.n_dimensions = len(self.parameter_nodes)
        self.model = model or GPyModel(self.n_dimensions, bounds=bounds,
                                       optimizer=optimizer, max_opt_iters=n_opt_iters)
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
            self.client = elfi_client()
            dask.set_options(get=self.client.get)

        if self.store is not None:
            if not isinstance(self.store, storage.NameIndexDataInterface):
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

