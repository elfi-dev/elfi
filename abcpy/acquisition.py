import numpy as np
import json

from scipy.stats import truncnorm

from .utils import stochastic_optimization, approx_second_partial_derivative

class AcquisitionBase():
    """ All acquisition functions are assumed to fulfill this interface """

    def __init__(self, model):
        self.model = model

    def _eval(self, x):
        """
            Evaluates the acquisition function value at 'x'
            type(x) = numpy.array
        """
        return 0.0

    def acquire(self, n_values):
        """
            Returns the next acquisition point.
            Return is of type <numpy.array>.

            type(n_values) = float
        """
        return numpy.zeros((int(n_values), self.model.n_var))


class LcbAcquisition(AcquisitionBase):

    def __init__(self, model, exploration_rate=2.0, opt_iterations=100):
        self.exploration_rate = float(exploration_rate)
        self.opt_iterations = int(opt_iterations)
        super(LcbAcquisition, self).__init__(model)

    def _eval(self, x):
        """ Lower confidence bound = mean - k * std """
        y_m, y_s2, y_s = self.model.evaluate(x)
        return float(y_m - self.exploration_rate * y_s)

    def acquire(self, n_values):
        minloc, val = stochastic_optimization(self._eval, self.model.bounds, self.opt_iterations)
        return np.atleast_2d([minloc] * int(n_values))


class SecondDerivativeNoiseMixin(AcquisitionBase):

    def __init__(self, *args, **kwargs):
        self.second_derivative_delta = float(kwargs.get("second_derivative_delta", 0.01))
        super(SecondDerivativeNoiseMixin, self).__init__(*args, **kwargs)

    def acquire(self, n_values):
        """ Adds noise based on function second derivative """
        opts = super(SecondDerivativeNoiseMixin, self).acquire(n_values)
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

class BolfiAcquisition(SecondDerivativeNoiseMixin, LcbAcquisition):
    pass
