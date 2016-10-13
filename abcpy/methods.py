import numpy as np

from .gpy_model import GpyModel
from .acquisition import BolfiAcquisition
from .posteriors import BolfiPosterior
from .utils import stochastic_optimization

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


class BOLFI(ABCMethod):

    def __init__(self, n_samples, distance_node=None, parameter_nodes=None, batch_size=10, model=None, acquisition=None, bounds=None, n_surrogate_samples=10):
        self.n_dimensions = len(parameter_nodes)
        self.model = model or GpyModel(self.n_dimensions, bounds)
        self.acquisition = acquisition or BolfiAcquisition(self.model)
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
        while self.model.n_observations() < self.n_surrogate_samples:
            locations = self.acquisition.acquire(self.batch_size)
            values_dict = {param.name: np.atleast_2d(locations[:,i]).T for i, param in enumerate(self.parameter_nodes)}
            values = self.distance_node.generate(len(locations), with_values=values_dict).compute()
            for i in range(len(locations)):
                print("Sample %d: %s at %s" % (self.model.n_observations()+i+1, values[i], locations[i]))
            self.model.update(locations, values)

    def getPosterior(self, threshold):
        return BolfiPosterior(self.model, threshold)

    def samplePosterior(self, threshold):
        posterior = self.getPosterior(threshold)
        return posterior.sample()

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
