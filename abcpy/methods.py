"""
These are sketches of how to use the ABC graphical model in the algorithms
"""
import numpy as np


class ABCMethod(object):
    def __init__(self, N, distance_node=None, parameter_nodes=None, batch_size=10):

        if not distance_node or not parameter_nodes:
            raise ValueError("Need to give the distance node and list of parameter nodes")

        self.N = N
        self.distance_node = distance_node
        self.parameter_nodes = parameter_nodes
        self.n_params = len(parameter_nodes)
        self.batch_size = batch_size

    def infer(self, spec, *args, **kwargs):
        raise NotImplementedError

    # Run the all-accepting sampler.
    def _get_distances(self, n_samples):

        distances = self.distance_node.generate(n_samples,
                    batch_size=self.batch_size).compute()
        parameters = [p.generate(n_samples).compute()
                      for p in self.parameter_nodes]

        return distances, parameters

    # Concatenate distances and parameters to existing.
    def _save_distances(self, distances, parameters):
        self.distances = np.concatenate((self.distances, distances), axis=0)
        self.parameters = [np.concatenate((self.parameters[ii], parameters[ii]), axis=0) for ii in range(self.n_params)]


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

        # only run at first call unless quantile specified
        if not hasattr(self, 'distances') or quantile:
            self.distances = np.empty((0,1))
            self.parameters = [ np.empty((0,1)) for ii in range(self.n_params) ]

            if quantile:
                distances, parameters = self._get_distances(int(self.N / quantile))
                threshold = np.percentile(distances, quantile*100)
                discard_rest = True
                save_values = True

            else:
                distances, parameters = self._get_distances(self.N)
                discard_rest = False
                save_values = True

        else:  # use precomputed distances
            distances = self.distances
            parameters = self.parameters
            discard_rest = False
            save_values = False

        posteriors = self.__apply_threshold(distances, parameters, threshold, discard_rest, save_values)

        return {'posteriors': posteriors, 'threshold': threshold}

    # Apply rejection criterion.
    def __apply_threshold(self, distances, parameters, threshold,
                          discard_rest=False, save_values=False):
        accepted = distances[:,0] < threshold
        posteriors = [p[accepted,:] for p in parameters]

        if discard_rest:
            distances = distances[accepted,:]
            parameters = posteriors

        if save_values:
            self._save_distances(distances, parameters)

        return posteriors


class BOLFI(ABCMethod):

    def infer(self, spec, parameters=None, distance=None, threshold=None):

        lik = GPLikelihoodApproximation().construct(parameters, distance)

        # TODO
        # - Construct PyMC model here using the lik
        # - Run the MCMC

        # Fixme: return the actual sample
        return lik


class GPLikelihoodApproximation():

    def construct(self, parameters=None, distance=None):

        while not self.GP.is_finished():
            values = self.acquisition.acquire()
            # Map the parameter values for the nodes
            values_hash = {param.name: values[:,i] for i, param in enumerate(parameters)}
            distances = distance.generate(len(values), self.batch_size, with_values=values_hash).compute()
            self.GP.update(parameters, distances)

        return self.GP












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
#             S = np.zeros([self.N, len(summaries)])
#             y = np.zeros([1, len(summaries)])
#             for i, s in enumerate(summaries):
#                 parameter_values[i].values[0:self.N] = params[i]
#                 S[:, i] = s.generate(self.N)
#                 y[i] = s.observed
#             cov = np.cov(S, rowvar=False)
#             mean = np.mean(S, axis=0)
#
#             lik = stats.multivariate_normal.pdf(y, mean, cov)
#             return lik
#
#         return objective
