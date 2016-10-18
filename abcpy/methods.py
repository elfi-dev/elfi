import numpy as np
from time import sleep
import dask
from distributed import Client

from .gpy_model import GpyModel
from .acquisition import LcbAcquisition, SecondDerivativeNoiseMixin, RbfAtPendingPointsMixin
from .utils import stochastic_optimization
from .async import wait

"""
These are sketches of how to use the ABC graphical model in the algorithms
"""
import numpy as np


class ABCMethod(object):
    def __init__(self, n_samples, distance_node=None, parameter_nodes=None, batch_size=10):

        if not distance_node or not parameter_nodes:
            raise ValueError("Need to give the distance node and list of parameter nodes")

        self.n_samples = n_samples
        self.distance_node = distance_node
        self.parameter_nodes = parameter_nodes
        self.batch_size = batch_size

    def infer(self, spec, *args, **kwargs):
        raise NotImplementedError


class Rejection(ABCMethod):
    """
    Rejection sampler.
    """
    def infer(self, threshold):
        """
        Run the rejection sampler. Inference can be repeated with a different
        threshold without rerunning the simulator.
        """

        # only run at first call
        if not hasattr(self, 'distances'):
            self.distances = self.distance_node.generate(self.n_samples, batch_size=self.batch_size).compute()
            self.parameters = [p.generate(self.n_samples, starting=0).compute()
                               for p in self.parameter_nodes]

        accepted = self.distances < threshold
        posteriors = [p[accepted] for p in self.parameters]

        return posteriors


class BolfiAcquisition(SecondDerivativeNoiseMixin, LcbAcquisition):
    pass


class AsyncBolfiAcquisition(SecondDerivativeNoiseMixin, RbfAtPendingPointsMixin, LcbAcquisition):
    pass


class BOLFI(ABCMethod):

    def __init__(self, n_samples, distance_node=None, parameter_nodes=None, batch_size=10, sync=True, model=None, acquisition=None, bounds=None, n_surrogate_samples=10):
        self.n_dimensions = len(parameter_nodes)
        self.model = model or GpyModel(self.n_dimensions, bounds)
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
        self.n_surrogate_samples = n_surrogate_samples
        super(BOLFI, self).__init__(n_samples, distance_node, parameter_nodes, batch_size)

    def infer(self, threshold=None):
        """
            Bolfi inference.

            type(threshold) = float
        """
        self.createSurrogate()
        return self.samplePosterior(threshold)

    def createSurrogate(self):
        print("Sampling %d samples in batches of %d" % (self.n_surrogate_samples, self.batch_size))
        all_values = None
        all_locations = None
        n_pending = 0
        client = Client()
        pending_indexes = list()
        ready_indexes = list()
        next_index = 0
        dask.set_options(get=client.get)
        while self.model.n_observations() < self.n_surrogate_samples:
            pending_locations = all_values[pending_indexes] if all_values is not None and len(pending_indexes) > 0 else None
            new_locations = self.acquisition.acquire(self.batch_size, pending_locations)
            new_values_dict = {param.name: np.atleast_2d(new_locations[:,i]).T for i, param in enumerate(self.parameter_nodes)}
            new_values = self.distance_node.generate(len(new_locations), with_values=new_values_dict)
            all_locations = np.vstack((all_locations, new_locations)) if all_locations is not None else new_locations
            all_values = all_values + new_values if all_values is not None else new_values
            if pending_locations is not None:
                pending_indexes = pending_indexes.extend(range(next_index, next_index + len(pending_locations)))
                next_index += max(max(pending_locations) + 1, next_index)
            new_ready_index = wait(list(all_values), client)  # TODO: add condition when wait() supports
            pending_indexes.remove(new_ready_index)
            ready_indexes.append(new_ready_index)
            self.model.update(np.atleast_2d(all_locations[new_ready_index]), np.atleast_2d(all_values[new_ready_index]))

    def getPosterior(self, threshold):
        return None

    def samplePosterior(self, threshold):
        return None

# class SyntheticLikelihood(ABCMethod):
#
#     def create_objective(self, model, parameters=None, summaries=None, **kwargs):
#         """
#
#         Parameters
#         ----------
#         model
#         parameter
#            array of nodes
#         summaries
#            array of nodes
#         kwargs
#
#         Returns
#         -------
#
#         """
#
#         parameter_values = []
#
#         for p in parameters:
#             values = Values()
#             values.replace(p, parents=False)
#             parameter_values.append(values)
#
#         def objective(params):
#             S = np.zeros([self.n_samples, len(summaries)])
#             y = np.zeros([1, len(summaries)])
#             for i, s in enumerate(summaries):
#                 parameter_values[i].values[0:self.n_samples] = params[i]
#                 S[:, i] = s.generate(self.n_samples)
#                 y[i] = s.observed
#             cov = np.cov(S, rowvar=False)
#             mean = np.mean(S, axis=0)
#
#             lik = stats.multivariate_normal.pdf(y, mean, cov)
#             return lik
#
#         return objective
