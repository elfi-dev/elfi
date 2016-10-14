import numpy as np
import json

from scipy.stats import truncnorm

from .utils import stochastic_optimization, approx_second_partial_derivative, sum_of_rbf_kernels

class AcquisitionBase():
    """ All acquisition functions are assumed to fulfill this interface """

    def __init__(self, model, n_max_parallel_values=None):
        self.model = model
        self.sync = True
        if n_max_parallel_values is not None:
            self.sync = False
            self.n_max_parallel_values = int(n_max_parallel_values)

    def _eval(self, x):
        """
            Evaluates the acquisition function value at 'x'

            type(x) = numpy.array
        """
        return 0.0

    def acquire(self, n_values, pending_locations=None):
        """
            Returns the next batch of acquisition points.
            Return is of type numpy.array_2d, locations on rows.

            Synchronous use:
            Returns 'n_values' samples.

            Asychronous use:
            Should keep total number of sample locations (number of values returned plus number
            of pending locations) less or equal to 'n_total_parallel_values'.

            type(n_values) = int
            type(n_new_values) = int
            type(pending_locations) = np.array_2d (pending locations on rows)
        """
        self.pending_locations = pending_locations
        if self.pending_locations is not None:
           self.n_pending_locations = pending_locations.shape[0]
        self.n_values = self._max_values_to_return(n_values)
        return np.zeros((n_values, self.model.n_var))

    def _max_values_to_return(self, n_values):
        if not self.sync:
            return min(int(n_values), int(n_total_parallel_values) - self.n_pending_locations)
        return n_values


class LcbAcquisition(AcquisitionBase):

    def __init__(self, model, exploration_rate=2.0, opt_iterations=100):
        self.exploration_rate = float(exploration_rate)
        self.opt_iterations = int(opt_iterations)
        super(LcbAcquisition, self).__init__(model)

    def _eval(self, x):
        """ Lower confidence bound = mean - k * std """
        y_m, y_s2, y_s = self.model.evaluate(x)
        return float(y_m - self.exploration_rate * y_s)

    def acquire(self, n_values, pending_locations=None):
        ret = super(LcbAcquisition, self).acquire(n_values, pending_locations)
        minloc, val = stochastic_optimization(self._eval, self.model.bounds, self.opt_iterations)
        for i in range(self.n_values):
            ret[i] = minloc
        return ret


class RbfAtPendingPointsMixin(AcquisitionBase):
    """ Adds RBF kernels at pending point locations """

    def __init__(self, *args, **kwargs):
        self.rbf_scale = float(kwargs.get("rbf_scale", 1.0))
        self.rbf_amplitude = float(kwargs.get("rbf_amplitude", 1.0))
        super(RbfAtPendingPointsMixin, self).__init__(*args, **kwargs)

    def _eval(self, x):
        val = super(RbfAtPendingPointsMixin, self)._eval(x)
        if self.pending_locations is None or self.pending_locations.shape[0] < 1:
            return val
        val += sum_of_rbf_kernels(x, self.pending_locations, self.rbf_amplitude, self.rbf_scale)
        return val


class SecondDerivativeNoiseMixin(AcquisitionBase):

    def __init__(self, *args, **kwargs):
        self.second_derivative_delta = float(kwargs.get("second_derivative_delta", 0.01))
        super(SecondDerivativeNoiseMixin, self).__init__(*args, **kwargs)

    def acquire(self, n_values, pending_locations=None):
        """ Adds noise based on function second derivative """
        opts = super(SecondDerivativeNoiseMixin, self).acquire(n_values, pending_locations)
        locs = list()
        for i in range(n_values):
            opt = opts[i]
            loc = list()
            for dim, val in enumerate(opt.tolist()):
                d2 = approx_second_partial_derivative(self._eval, opt, dim,
                        self.second_derivative_delta, self.model.bounds)
                # std from mathching second derivative to that of normal
                # -N(0,std)'' = 1/(sqrt(2pi)std^3) = der2
                # => std = der2 ** -1/3 * (2*pi) ** -1/6
                if d2 > 0:
                    std = np.power(2*np.pi, -1.0/6.0) * np.power(d2, -1.0/3.0)
                else:
                    std = float("inf")
                low = self.model.bounds[i][0]
                high = self.model.bounds[i][1]
                maxstd = (high - low) / 2.0
                std = min(std, maxstd)  # limit noise amount based on bounds
                newval = truncnorm.rvs(low, high, loc=val, scale=std)
                loc.append(newval)
            locs.append(loc)
        return np.atleast_2d(locs)



