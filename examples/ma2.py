from functools import partial
import numpy as np
import scipy.stats as ss
import elfi
from elfi.model.extensions import ScipyLikeDistribution

"""Example implementation of the MA2 model
"""


def MA2(t1, t2, n_obs=100, batch_size=1, random_state=None, w=None):
    if w is None:
        if random_state is None:
            # Use the global random_state
            random_state = np.random
        w = random_state.randn(batch_size, n_obs+2) # i.i.d. sequence ~ N(0,1)

    w = np.atleast_2d(w)
    x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
    return x


def autocov(x, lag=1):
    """Autocovariance assuming a (weak) univariate stationary process with mean 0.
    Realizations are in rows.
    """
    C = np.mean(x[:, lag:]*x[:, :-lag], axis=1, keepdims=True)
    return C


def discrepancy(x, y):
    d = np.linalg.norm(np.array(x) - np.array(y), ord=2, axis=0)
    return d


def get_model(n_obs=100, true_params=None, seed_obs=None):
    """Returns a complete MA2 model in inference task

    Parameters
    ----------
    n_obs : int
        observation length of the MA2 process
    true_params : list
        parameters with which the observed data is generated
    seed_obs : None, int
        seed for the observed data generation

    Returns
    -------
    InferenceTask
    """
    if true_params is None:
        true_params = [.6, .2]

    y = MA2(*true_params, n_obs=n_obs, random_state=np.random.RandomState(seed_obs))
    sim_fn = partial(MA2, n_obs=n_obs)

    m = elfi.ElfiModel()

    t1 = elfi.Prior('t1', CustomPrior1, 2, model=m)
    t2 = elfi.Prior('t2', CustomPrior2, t1, 1, model=m)
    Y = elfi.Simulator('MA2', sim_fn, t1, t2, observed=y, model=m)
    S1 = elfi.Summary('S1', autocov, Y, model=m)
    S2 = elfi.Summary('S2', autocov, Y, 2, model=m)
    d = elfi.Discrepancy('d', discrepancy, S1, S2, model=m)

    return m


# Define prior t1 as in Marin et al., 2012 with t1 in range [-b, b]
class CustomPrior1(ScipyLikeDistribution):
    @classmethod
    def rvs(cls, b, size=1, random_state=None):
        u = ss.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
        t1 = np.where(u < 0.5, np.sqrt(2.*u)*b - b, -np.sqrt(2.*(1. - u))*b + b)
        return t1


# Define prior t2 conditionally on t1 as in Marin et al., 2012, in range [-a, a]
class CustomPrior2(ScipyLikeDistribution):
    @classmethod
    def rvs(cls, t1, a, size=1, random_state=None):
        """

        Parameters
        ----------
        t1 : float
        a  : float
        size : int or tuple
        random_state : None, RandomState

        Returns
        -------

        """

        locs = np.maximum(-a - t1, -a + t1)
        scales = a - locs
        t2 = ss.uniform.rvs(loc=locs, scale=scales, size=size, random_state=random_state)
        return t2
