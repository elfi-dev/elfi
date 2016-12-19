from functools import partial
import numpy as np
import elfi

"""Example implementation of the MA2 model
"""


# FIXME: move n_obs as kw argument. Change the simulator interface accordingly
def MA2(n_obs, t1, t2, batch_size=1, random_state=None, latents=None):
    if latents is None:
        if random_state is None:
            random_state = np.random.RandomState()
        latents = random_state.randn(batch_size, n_obs+2) # i.i.d. sequence ~ N(0,1)
    u = np.atleast_2d(latents)
    y = u[:,2:] + t1 * u[:,1:-1] + t2 * u[:,:-2]
    return y


def autocov(x, lag=1):
    """Autocovariance assuming a (weak) univariate stationary process with mean 0.
    Realizations are in rows.
    """

    # To avoid deprecation warning when `lag.ndim` > 0. Happens if lag is acquired from a
    # Constant operation node.
    lag = np.squeeze(lag)
    C = np.mean(x[:,lag:] * x[:,:-lag], axis=1, keepdims=True)
    return C


def discrepancy(x, y):
    d = np.linalg.norm( np.array(x) - np.array(y), ord=2, axis=0)
    return d


def inference_task(n_obs=100, params_obs=None, seed_obs=12345):
    """Returns a complete MA2 model in inference task

    Parameters
    ----------
    n_obs : observation length of the MA2 process
    params_obs : parameters with which the observed data is generated
    seed_obs : seed for the observed data generation

    Returns
    -------
    InferenceTask
    """
    if params_obs is None:
        params_obs = [.6, .2]
    if len(params_obs) != 2:
        raise ValueError("Invalid length of params_obs. Should be 2.")

    y = MA2(n_obs, *params_obs, random_state=np.random.RandomState(seed_obs))
    sim = partial(MA2, n_obs)
    itask = elfi.InferenceTask()
    t1 = elfi.Prior('t1', 'uniform', 0, 1, inference_task=itask)
    t2 = elfi.Prior('t2', 'uniform', 0, 1, inference_task=itask)
    Y = elfi.Simulator('MA2', sim, t1, t2, observed=y, inference_task=itask)
    S1 = elfi.Summary('S1', autocov, Y, inference_task=itask)
    S2 = elfi.Summary('S2', autocov, Y, 2, inference_task=itask)
    d = elfi.Discrepancy('d', discrepancy, S1, S2, inference_task=itask)
    itask.parameters = [t1, t2]
    return itask
