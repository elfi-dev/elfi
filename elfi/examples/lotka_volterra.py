"""Example of inference with the stochastic Lotka-Volterra predator prey model.

Treatment roughly follows:
Owen, J., Wilkinson, D. and Gillespie, C. (2015) Likelihood free inference for
   Markov processes: a comparison. Statistical Applications in Genetics and
   Molecular Biology, 14(2).
"""

import logging
from functools import partial

import numpy as np
import scipy.stats as ss

import elfi


def lotka_volterra(r1, r2, r3, prey_init=50, predator_init=100, sigma=0., n_obs=16, time_end=30.,
                   batch_size=1, random_state=None, return_full=False):
    r"""Generate sequences from the stochastic Lotka-Volterra model.

    The Lotka-Volterra model is described by 3 reactions

    R1 : X1 -> 2X1       # prey reproduction
    R2 : X1 + X2 -> 2X2  # predator hunts prey and reproduces
    R3 : X2 -> 0         # predator death

    The system is solved using the Direct method.

    Gillespie, D. T. (1977) Exact stochastic simulation of coupled chemical reactions.
        The Journal of Physical Chemistry 81 (25), 2340–2361.
    Lotka, A. J. (1925) Elements of physical biology. Williams & Wilkins Baltimore.
    Volterra, V. (1926) Fluctuations in the abundance of a species considered mathematically.
        Nature 118, 558–560.


    Parameters
    ----------
    r1 : float or np.array
        Rate of R1.
    r2 : float or np.array
        Rate of R2.
    r3 : float or np.array
        Rate of R3.
    prey_init : int or np.array, optional
        Initial number of prey.
    predator_init : int or np.array, optional
        Initial number of predators.
    sigma : float or np.array, optional
        Standard deviation of the Gaussian noise added to measurements.
    n_obs : int, optional
        Number of observations to return at integer frequency.
    time_end : float, optional
        Time allowed for reactions in the Direct method (not the wall time).
    batch_size : int, optional
    random_state : np.random.RandomState, optional
    return_full : bool, optional
        If True, return a tuple (observed_stock, observed_times, full_stock, full_times).

    Returns
    -------
    stock_obs : np.array
        Observations in shape (batch_size, n_obs, 2).

    """
    random_state = random_state or np.random

    r1 = np.asanyarray(r1).reshape(-1)
    r2 = np.asanyarray(r2).reshape(-1)
    r3 = np.asanyarray(r3).reshape(-1)
    prey_init = np.asanyarray(prey_init).reshape(-1)
    predator_init = np.asanyarray(predator_init).reshape(-1)
    sigma = np.asanyarray(sigma).reshape(-1)

    n_full = 20000
    stock = np.empty((batch_size, n_full, 2), dtype=np.int32)
    # As we use approximate continuous priors for prey_init and
    # predator_init, we'll round them down to closest integers
    stock[:, 0, 0] = np.floor(prey_init)
    stock[:, 0, 1] = np.floor(predator_init)
    stoichiometry = np.array([[1, 0], [-1, 1], [0, -1], [0, 0]], dtype=np.int32)
    times = np.empty((batch_size, n_full))
    times[:, 0] = 0

    # iterate until all in batch ok
    ii = 0
    while np.any(times[:, ii] < time_end):
        ii += 1

        # increase the size of arrays if needed
        if ii == n_full:
            stock = np.concatenate((stock, np.empty((batch_size, n_full, 2))), axis=1)
            times = np.concatenate((times, np.empty((batch_size, n_full))), axis=1)
            n_full *= 2

        # reaction probabilities
        hazards = np.column_stack((r1 * stock[:, ii - 1, 0],
                                   r2 * stock[:, ii - 1, 0] * stock[:, ii - 1, 1],
                                   r3 * stock[:, ii - 1, 1]))

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sum_hazards = 1. / np.sum(hazards, axis=1, keepdims=True)  # inf if all dead

            delta_t = random_state.exponential(inv_sum_hazards.ravel())
            times[:, ii] = times[:, ii - 1] + delta_t

            # choose reaction according to their probabilities
            probs = hazards * inv_sum_hazards
            cumprobs = np.cumsum(probs[:, :-1], axis=1)
            x = random_state.uniform(size=(batch_size, 1))
            reaction = np.sum(x >= cumprobs, axis=1)

        # null reaction if both populations dead
        reaction = np.where(np.isinf(inv_sum_hazards.ravel()), 3, reaction)

        # update stock
        stock[:, ii, :] = stock[:, ii - 1, :] + stoichiometry[reaction, :]

        # no point to continue if predators = 0
        times[:, ii] = np.where(stock[:, ii, 1] == 0, time_end, times[:, ii])

    stock = stock[:, :ii + 1, :]
    times = times[:, :ii + 1]

    times_out = np.linspace(0, time_end, n_obs)
    stock_out = np.empty((batch_size, n_obs, 2), dtype=np.int32)
    stock_out[:, 0, :] = stock[:, 0, :]

    # observations at even intervals
    for ii in range(1, n_obs):
        iy, ix = np.where(times >= times_out[ii])
        iy, iix = np.unique(iy, return_index=True)
        ix = ix[iix] - 1
        time_term = (times_out[ii] - times[iy, ix]) / (times[iy, ix + 1] - times[iy, ix])
        stock_out[:, ii, 0] = (stock[iy, ix + 1, 0] - stock[iy, ix, 0]) * time_term \
            + stock[iy, ix, 0] + random_state.normal(scale=sigma, size=batch_size)
        stock_out[:, ii, 1] = (stock[iy, ix + 1, 1] - stock[iy, ix, 1]) * time_term \
            + stock[iy, ix, 1] + random_state.normal(scale=sigma, size=batch_size)

    if return_full:
        return (stock_out, times_out, stock, times)

    return stock_out


def get_model(n_obs=50, true_params=None, observation_noise=False, seed_obs=None, **kwargs):
    """Return a complete Lotka-Volterra model in inference task.

    Including observation noise to system is optional.

    Parameters
    ----------
    n_obs : int, optional
        Number of observations.
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.
    observation_noise : bool, optional
        Whether or not add normal noise to observations.

    Returns
    -------
    m : elfi.ElfiModel

    """
    logger = logging.getLogger()
    if true_params is None:
        if observation_noise:
            true_params = [1.0, 0.005, 0.6, 50, 100, 10.]
        else:
            true_params = [1.0, 0.005, 0.6, 50, 100, 0.]
    else:
        if observation_noise:
            if len(true_params) != 6:
                raise ValueError(
                        "Option observation_noise = True."
                        " Provide six input parameters."
                        )
        else:
            if len(true_params) != 5:
                raise ValueError(
                        "Option observation_noise = False."
                        " Provide five input parameters."
                        )
            true_params = true_params + [0]

    kwargs['n_obs'] = n_obs
    y_obs = lotka_volterra(*true_params, random_state=np.random.RandomState(seed_obs), **kwargs)

    m = elfi.ElfiModel()
    sim_fn = partial(lotka_volterra, **kwargs)
    priors = [
        elfi.Prior(ExpUniform, -6., 2., model=m, name='r1'),
        elfi.Prior(ExpUniform, -6., 2., model=m, name='r2'),  # easily kills populations
        elfi.Prior(ExpUniform, -6., 2., model=m, name='r3'),
        elfi.Prior('normal', 50, np.sqrt(50), model=m, name='prey0'),
        elfi.Prior('normal', 100, np.sqrt(100), model=m, name='predator0')
    ]

    if observation_noise:
        priors.append(elfi.Prior(ExpUniform, np.log(0.5), np.log(50), model=m, name='sigma'))

    elfi.Simulator(sim_fn, *priors, observed=y_obs, name='LV')

    sumstats = [
        elfi.Summary(partial(stock_mean, species=0), m['LV'], name='prey_mean'),
        elfi.Summary(partial(stock_mean, species=1), m['LV'], name='pred_mean'),
        elfi.Summary(partial(stock_log_variance, species=0), m['LV'], name='prey_log_var'),
        elfi.Summary(partial(stock_log_variance, species=1), m['LV'], name='pred_log_var'),
        elfi.Summary(partial(stock_autocorr, species=0, lag=1), m['LV'], name='prey_autocorr_1'),
        elfi.Summary(partial(stock_autocorr, species=1, lag=1), m['LV'], name='pred_autocorr_1'),
        elfi.Summary(partial(stock_autocorr, species=0, lag=2), m['LV'], name='prey_autocorr_2'),
        elfi.Summary(partial(stock_autocorr, species=1, lag=2), m['LV'], name='pred_autocorr_2'),
        elfi.Summary(stock_crosscorr, m['LV'], name='crosscorr')
    ]

    elfi.Distance('euclidean', *sumstats, name='d')

    logger.info("Generated %i observations with true parameters r1: %.1f, r2: %.3f, r3: %.1f, "
                "prey0: %i, predator0: %i, sigma: %.1f.", n_obs, *true_params)

    return m


def stock_mean(stock, species=0, mu=0, std=1):
    """Calculate the mean of the trajectory by species."""
    stock = np.atleast_2d(stock[:, :, species])
    mu_x = np.mean(stock, axis=1)

    return (mu_x - mu) / std


def stock_log_variance(stock, species=0, mu=0, std=1):
    """Calculate the log variance of the trajectory by species."""
    stock = np.atleast_2d(stock[:, :, species])
    var_x = np.var(stock, axis=1, ddof=1)
    log_x = np.log(var_x + 1)

    return (log_x - mu) / std


def stock_autocorr(stock, species=0, lag=1, mu=0, std=1):
    """Calculate the autocorrelation of lag n of the trajectory by species."""
    stock = np.atleast_2d(stock[:, :, species])
    n_obs = stock.shape[1]

    mu_x = np.mean(stock, axis=1, keepdims=True)
    std_x = np.std(stock, axis=1, ddof=1, keepdims=True)
    sx = ((stock - np.repeat(mu_x, n_obs, axis=1)) / np.repeat(std_x, n_obs, axis=1))
    sx_t = sx[:, lag:]
    sx_s = sx[:, :-lag]

    C = np.sum(sx_t * sx_s, axis=1) / (n_obs - 1)

    return (C - mu) / std


def stock_crosscorr(stock, mu=0, std=1):
    """Calculate the cross correlation of the species trajectories."""
    n_obs = stock.shape[1]

    x_preys = stock[:, :, 0]  # preys
    x_preds = stock[:, :, 1]  # predators

    mu_preys = np.mean(x_preys, axis=1, keepdims=True)
    mu_preds = np.mean(x_preds, axis=1, keepdims=True)
    std_preys = np.std(x_preys, axis=1, keepdims=True)
    std_preds = np.std(x_preds, axis=1, keepdims=True)
    s_preys = ((x_preys - np.repeat(mu_preys, n_obs, axis=1))
               / np.repeat(std_preys, n_obs, axis=1))
    s_preds = ((x_preds - np.repeat(mu_preds, n_obs, axis=1))
               / np.repeat(std_preds, n_obs, axis=1))

    C = np.sum(s_preys * s_preds, axis=1) / (n_obs - 1)

    return (C - mu) / std


class ExpUniform(elfi.Distribution):
    r"""Prior distribution for parameter.

    log x ~ Uniform(a,b)
    pdf(x) \propto 1/x, if x \in [exp(a), exp(b)]

    """

    @classmethod
    def rvs(cls, a, b, size=1, random_state=None):
        """Draw random variates.

        Parameters
        ----------
        a : float or array-like
        b : float or array-like
        size : int, optional
        random_state : RandomState, optional

        Returns
        -------
        np.array

        """
        u = ss.uniform.rvs(loc=a, scale=b - a, size=size, random_state=random_state)
        x = np.exp(u)
        return x

    @classmethod
    def pdf(cls, x, a, b):
        """Density function at `x`.

        Parameters
        ----------
        x : float or array-like
        a : float or array-like
        b : float or array-like

        Returns
        -------
        np.array

        """
        with np.errstate(divide='ignore'):
            p = np.where((x < np.exp(a)) | (x > np.exp(b)), 0, np.reciprocal(x))
            p /= (b - a)  # normalize
        return p
