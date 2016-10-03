"""
These are sketches of how to use the ABC graphical model in the algorithms
"""

class ABCMethod(object):
    def __init__(self, N, batch_size=10):
        self.N = N
        self.batch_size = batch_size

    def infer(self, spec, *args, **kwargs):
        raise NotImplementedError


class Rejection(ABCMethod):

    def infer(self, spec, parameters=None, threshold=None):

        thresholds = threshold.generate(self.N, self.batch_size)
        params = [param.generate(self.N, self.batch_size) for param in parameters]
        return [param[thresholds] for param in params]


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
