import sys
import logging
import json
import numpy as np

from scipy.stats import truncnorm

from .utils import approx_second_partial_derivative, sum_of_rbf_kernels
from ..utils import stochastic_optimization

logger = logging.getLogger(__name__)

"""Implementations of some acquisition functions.

The acquisition function defines the policy for choosing
the points where to sample in Bayesian optimization.

AcquisitionBase     : Base class for acquisition functions
AcquisitionSchedule : A sequence of acquisition functions
LCBAcquisition      : Lower Confidence Bound acquisition
RbfAtPendingPointsMixin    : Adds a RBF kernel to pending points (async batch)
SecondDerivativeNoiseMixin : Adds noise based on second derivative estimate (batch)
"""

class AcquisitionBase():
    """All acquisition functions are assumed to fulfill this interface.

    Parameters
    ----------
    model : an object with attributes
                input_dim : int
                bounds : tuple of length 'input_dim' of tuples (min, max)
            and methods
                evaluate(x) : function that returns model (mean, var, std)
    n_samples : None or int
        Total number of samples to be sampled, used when part of an
        AcquisitionSchedule object (None indicates no upper bound)
    """
    def __init__(self, model, n_samples=None):
        self.model = model
        self.n_samples = n_samples
        self.n_acquired = 0

    def __add__(self, acquisition):
        return AcquisitionSchedule(schedule=[self, acquisition])

    def _eval(self, x):
        """Evaluates the acquisition function value at 'x'

        Returns
        -------
        x : numpy.array
        """
        return NotImplementedError

    def acquire(self, n_values, pending_locations=None):
        """Returns the next batch of acquisition points.

        Parameters
        ----------
        n_values : int
            Number of values to return.
        pending_locations : None or numpy 2d array
            If given, asycnhronous acquisition functions may
            use the locations in choosing the next sampling
            location. Locations should be in rows.

        Returns
        -------
        locations : 2D np.ndarray of shape (n_values, ...)
        """
        self.pending_locations = pending_locations
        if self.pending_locations is not None:
           self.n_pending_locations = pending_locations.shape[0]
        self.n_values = n_values
        self.n_acquired += n_values
        return np.zeros((n_values, self.model.input_dim))

    @property
    def samples_left(self):
        """Number of samples left to sample or sys.maxsize if no limit.
        """
        if self.n_samples is None:
            return sys.maxsize
        return self.n_samples - self.n_acquired

    @property
    def finished(self):
        """False if number of acquired samples is less than total samples.
        """
        return self.samples_left < 1


class AcquisitionSchedule(AcquisitionBase):
    """A sequence of acquisition functions.

    Parameters
    ----------
    schedule : list of AcquisitionBase objects
    """

    def __init__(self, schedule=None):
        if schedule is None or len(schedule) < 1:
            raise ValueError("Schedule must contain at least one element.")
        self.schedule = schedule
        self._check_schedule()

    def _check_schedule(self):
        """Raises an error if schedule is not valid.

        All acquisition functions should inherit AcquisitionBase.
        All acquisition functions should share the same model.
        All acquisition functions in the schedule should be reachable.
        """
        model = self.schedule[0].model
        at_end = False
        for acq in self.schedule:
            if not isinstance(acq, AcquisitionBase):
                raise ValueError("Only AcquisitionBase objects can be added to the schedule.")
            if at_end is True:
                raise ValueError("Unreachable acquisition function at the end of list.")
            if acq.n_samples is None:
                at_end = True

    def __add__(self, acquisition):
        """Appends an acquisition function to the schedule.
        """
        self.schedule.append(acquisition)
        self._check_schedule()
        return self

    def _get_next(self):
        """Returns next acquisition function in schedule.
        """
        for acq in self.schedule:
            if not acq.finished:
                return acq
        return None

    def acquire(self, n_values, pending_locations=None):
        """Returns the next batch of acquisition points.

        Parameters
        ----------
        n_values : int
            Number of values to return.
        pending_locations : None or numpy 2d array
            If given, asycnhronous acquisition functions may
            use the locations in choosing the next sampling
            location. Locations should be in rows.

        Returns
        -------
        Return is of type numpy.array_2d, locations on rows.
        """
        acq = self._get_next()
        if acq is None:
            raise IndexError("No more acquisition functions in schedule")
        if n_values > acq.samples_left:
            raise NotImplementedError("Acquisition function number of samples must be "
                    "multiple of n_values. Dividing a batch to multiple acquisition "
                    "functions is not yet implemented.")
        return acq.acquire(n_values, pending_locations)

    @property
    def samples_left(self):
        """ Return number of samples left to sample or sys.maxsize if no limit """
        s_left = 0
        for acq in self.schedule:
            if acq.n_samples is not None:
                s_left += acq.samples_left
            else:
                return sys.maxsize
        return s_left

    @property
    def finished(self):
        """ Returns False if number of acquired samples is less than
            number of total samples
        """
        return self.samples_left < 1


class LCBAcquisition(AcquisitionBase):

    def __init__(self, *args, exploration_rate=2.0, opt_iterations=100, **kwargs):
        self.exploration_rate = float(exploration_rate)
        self.opt_iterations = int(opt_iterations)
        super(LCBAcquisition, self).__init__(*args, **kwargs)

    def _eval(self, x):
        """ Lower confidence bound = mean - k * std """
        y_m, y_s2, y_s = self.model.evaluate(x)
        return float(y_m - self.exploration_rate * y_s)

    def acquire(self, n_values, pending_locations=None):
        ret = super(LCBAcquisition, self).acquire(n_values, pending_locations)
        minloc, val = stochastic_optimization(self._eval, self.model.bounds, self.opt_iterations)
        for i in range(self.n_values):
            ret[i] = minloc
        return ret


class RandomAcquisition(AcquisitionBase):
    """Acquisition purely from priors. This can be useful if parameters
    in certain regions are forbidden (i.e. their pdf is zero).

    Parameters
    ----------
    prior_list : list of Prior objects

    """

    def __init__(self, prior_list, *args, **kwargs):
        self.prior_list = prior_list
        n_priors = len(prior_list)

        # hacky...
        class DummyModel(object):
            pass
        model = DummyModel()
        model.input_dim = n_priors
        model.bounds = tuple(zip([0]*n_priors, [1]*n_priors))
        model.evaluate = lambda x : exec('raise NotImplementedError')

        super(RandomAcquisition, self).__init__(*args, model=model, **kwargs)

    def acquire(self, n_values, pending_locations=None):
        ret = super(RandomAcquisition, self).acquire(n_values, pending_locations)
        for i, p in enumerate(self.prior_list):
            ret[:, i] = p.generate(n_values).compute().ravel()
        logger.debug("Acquired {}".format(n_values))
        return ret


class RbfAtPendingPointsMixin(AcquisitionBase):
    """ Adds RBF kernels at pending point locations """

    def __init__(self, *args, rbf_scale=1.0, rbf_amplitude=1.0, **kwargs):
        self.rbf_scale = rbf_scale
        self.rbf_amplitude = rbf_amplitude
        super(RbfAtPendingPointsMixin, self).__init__(*args, **kwargs)

    def _eval(self, x):
        val = super(RbfAtPendingPointsMixin, self)._eval(x)
        if self.pending_locations is None or self.pending_locations.shape[0] < 1:
            return val
        val += sum_of_rbf_kernels(x, self.pending_locations, self.rbf_amplitude, self.rbf_scale)
        return val


class SecondDerivativeNoiseMixin(AcquisitionBase):

    def __init__(self, *args, second_derivative_delta=0.01, **kwargs):
        self.second_derivative_delta = second_derivative_delta
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
                low = self.model.bounds[dim][0]
                high = self.model.bounds[dim][1]
                maxstd = (high - low) / 2.0
                std = min(std, maxstd)  # limit noise amount based on bounds
                a, b = (low - val) / std, (high - val) / std  # standard bounds
                newval = truncnorm.rvs(a, b, loc=val, scale=std)
                loc.append(newval)
            locs.append(loc)
        return np.atleast_2d(locs)

