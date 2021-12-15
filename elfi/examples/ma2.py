"""Example implementation of the 2nd order Moving Average (MA2) model."""

from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def MA2(t1, t2, n_obs=100, batch_size=1, random_state=None):
    r"""Generate a sequence of samples from the MA2 model.

    The sequence is a moving average

        x_i = w_i + \theta_1 w_{i-1} + \theta_2 w_{i-2}

    where w_i are white noise ~ N(0,1).

    Parameters
    ----------
    t1 : float, array_like
    t2 : float, array_like
    n_obs : int, optional
    batch_size : int, optional
    random_state : RandomState, optional

    """
    # Make inputs 2d arrays for broadcasting with w
    t1 = np.asanyarray(t1).reshape((-1, 1))
    t2 = np.asanyarray(t2).reshape((-1, 1))
    random_state = random_state or np.random

    # i.i.d. sequence ~ N(0,1)
    w = random_state.randn(batch_size, n_obs + 2)
    x = w[:, 2:] + t1 * w[:, 1:-1] + t2 * w[:, :-2]
    return x


def autocov(x, lag=1):
    """Return the autocovariance.

    Assumes a (weak) univariate stationary process with mean 0.
    Realizations are in rows.

    Parameters
    ----------
    x : np.array of size (n, m)
    lag : int, optional

    Returns
    -------
    C : np.array of size (n,)

    """
    x = np.atleast_2d(x)
    # In R this is normalized with x.shape[1]
    C = np.mean(x[:, lag:] * x[:, :-lag], axis=1)
    return C


def get_model(n_obs=100, true_params=None, seed_obs=None):
    """Return a complete MA2 model in inference task.

    Parameters
    ----------
    n_obs : int, optional
        observation length of the MA2 process
    true_params : list, optional
        parameters with which the observed data is generated
    seed_obs : int, optional
        seed for the observed data generation

    Returns
    -------
    m : elfi.ElfiModel

    """
    if true_params is None:
        true_params = [.6, .2]

    y = MA2(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(MA2, n_obs=n_obs)

    m = elfi.ElfiModel()
    elfi.Prior(CustomPrior1, 2, model=m, name='t1')
    elfi.Prior(CustomPrior2, m['t1'], 1, name='t2')
    elfi.Simulator(sim_fn, m['t1'], m['t2'], observed=y, name='MA2')
    elfi.Summary(autocov, m['MA2'], name='S1')
    elfi.Summary(autocov, m['MA2'], 2, name='S2')
    elfi.Distance('euclidean', m['S1'], m['S2'], name='d')
    return m


# Define prior t1 as in Marin et al., 2012 with t1 in range [-b, b]
class CustomPrior1(elfi.Distribution):
    """Define prior for t1 in range [-a, a].

    As in Marin et al., 2012.
    """

    @classmethod
    def rvs(cls, b, size=1, random_state=None):
        """Get random variates.

        Parameters
        ----------
        b : float
        size : int or tuple, optional
        random_state : RandomState, optional

        Returns
        -------
        arraylike

        """
        u = ss.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
        t1 = np.where(u < 0.5, np.sqrt(2. * u) * b - b, -np.sqrt(2. * (1. - u)) * b + b)
        return t1

    @classmethod
    def pdf(cls, x, b):
        """Return density at `x`.

        Parameters
        ----------
        x : float or arraylike
        b : float

        Returns
        -------
        arraylike

        """
        p = 1. / b - np.abs(x) / (b * b)
        # set values outside of [-b, b] to zero
        p = np.where(p < 0., 0., p)
        return p


class CustomPrior2(elfi.Distribution):
    """Define prior for t2 conditionally on t1 in range [-a, a].

    As in Marin et al., 2012.
    """

    @classmethod
    def rvs(cls, t1, a, size=1, random_state=None):
        """Get random variates.

        Parameters
        ----------
        t1 : float or arraylike
        a : float
        size : int or tuple, optional
        random_state : RandomState, optional

        Returns
        -------
        arraylike

        """
        locs = np.maximum(-a - t1, -a + t1)
        scales = a - locs
        t2 = ss.uniform.rvs(loc=locs, scale=scales, size=size, random_state=random_state)
        return t2

    @classmethod
    def pdf(cls, x, t1, a):
        """Return density at `x`.

        Parameters
        ----------
        x : float or arraylike
        t1 : float or arraylike
        a : float

        Returns
        -------
        arraylike

        """
        locs = np.maximum(-a - t1, -a + t1)
        scales = a - locs
        p = (x >= locs) * (x <= locs + scales) * 1 / np.where(scales > 0, scales, 1)
        return p
