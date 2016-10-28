import numpy as np
from time import sleep
import dask
from distributed import Client

from elfi.bo.gpy_model import GpyModel
from elfi.bo.acquisition import LcbAcquisition, SecondDerivativeNoiseMixin, RbfAtPendingPointsMixin
from elfi.utils import stochastic_optimization
from elfi.posteriors import BolfiPosterior
from .async import wait

"""
These are sketches of how to use the ABC graphical model in the algorithms
"""

class ABCMethod(object):
    def __init__(self, n_samples, distance_node=None, parameter_nodes=None, batch_size=10):

        if distance_node is None or parameter_nodes is None:
            raise ValueError("Need to give the distance node and list of parameter nodes")

        self.n_samples = n_samples
        self.distance_node = distance_node
        self.parameter_nodes = parameter_nodes
        self.n_params = len(parameter_nodes)
        self.batch_size = batch_size

    def infer(self, spec, *args, **kwargs):
        raise NotImplementedError

    # Run the all-accepting sampler.
    def _get_distances(self, n_samples):

        distances = self.distance_node.acquire(n_samples, batch_size=self.batch_size).compute()
        parameters = [p.acquire(n_samples, batch_size=self.batch_size).compute()
                      for p in self.parameter_nodes]

        return distances, parameters


class Rejection(ABCMethod):
    """
    Rejection sampler.
    """
    def infer(self, threshold=None, quantile=None):
        """
        Run the rejection sampler. Inference can be repeated with a different
        threshold without rerunning the simulator.
        - 1st run with threshold: run simulator and apply threshold
        - next runs with threshold: only apply threshold to existing distances
        - all runs with quantile: run simulator
        """

        n_samples = self.n_samples if quantile is None else int(self.n_samples / quantile)

        distances, parameters = self._get_distances(n_samples)
        distances = distances.ravel()  # avoid unnecessary indexing

        if quantile is not None:    # filter with quantile
            sorted_inds = np.argsort(distances)
            threshold = distances[ sorted_inds[self.n_samples-1] ]
            accepted = sorted_inds[:self.n_samples]

        else:  # filter with predefined threshold
            accepted = distances < threshold

        posteriors = [p[accepted] for p in parameters]

        return {'samples': posteriors, 'threshold': threshold}


class BolfiAcquisition(SecondDerivativeNoiseMixin, LcbAcquisition):
    pass


class AsyncBolfiAcquisition(SecondDerivativeNoiseMixin, RbfAtPendingPointsMixin, LcbAcquisition):
    pass


class BOLFI(ABCMethod):

    def __init__(self, n_samples, distance_node=None, parameter_nodes=None, batch_size=10, sync=True, model=None, acquisition=None, bounds=None, n_surrogate_samples=10):
        self.n_dimensions = len(parameter_nodes)
        self.model = model or GpyModel(self.n_dimensions, bounds=bounds)
        self.sync = sync
        if acquisition is not None:
            self.acquisition = acquisition
            self.sync = self.acquisition.sync
        elif sync is True:
            self.acquisition = BolfiAcquisition(self.model)
        else:
            self.acquisition = AsyncBolfiAcquisition(self.model, batch_size)
        if self.sync is True:
            self.sync_condition = "all"
        else:
            self.sync_condition = "any"
        from distributed import Client
        self.client = Client()
        dask.set_options(get=self.client.get)
        self.n_surrogate_samples = n_surrogate_samples
        super(BOLFI, self).__init__(n_samples, distance_node, parameter_nodes, batch_size)

    def infer(self, threshold=None):
        """Bolfi inference.

        Parameters
        ----------
            threshold: float
        """
        self.create_surrogate_likelihood()
        return self.get_posterior(threshold)

    def create_surrogate_likelihood(self):
        if self.sync is True:
            print("Sampling %d samples in batches of %d" % (self.n_surrogate_samples, self.batch_size))
        else:
            print("Sampling %d samples asynchronously %d samples in parallel" % (self.n_surrogate_samples, self.batch_size))
        futures = list()  # pending future results
        pending = list()  # pending locations matched to futures by list index
        while self.model.n_observations() < self.n_surrogate_samples:
            next_batch_size = self._next_batch_size(len(pending))
            if next_batch_size > 0:
                pending_locations = np.atleast_2d(pending) if len(pending) > 0 else None
                new_locations = self.acquisition.acquire(next_batch_size, pending_locations)
                for location in new_locations:
                    wv_dict = {param.name: np.atleast_2d(location[i]) for i, param in enumerate(self.parameter_nodes)}
                    future = self.distance_node.generate(1, with_values=wv_dict)
                    futures.append(future)
                    pending.append(location)
            result, result_index, futures = wait(futures, self.client)
            location = pending.pop(result_index)
            self.model.update(location, result)

    def _next_batch_size(self, n_pending):
        if self.sync is True and n_pending > 0:
            return 0
        samples_left = self.n_surrogate_samples - self.model.n_observations()
        return min(self.batch_size, samples_left) - n_pending

    def get_posterior(self, threshold):
        return BolfiPosterior(self.model, threshold)


