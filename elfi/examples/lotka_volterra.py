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
    stock[:, 0, 0] = prey_init
    stock[:, 0, 1] = predator_init
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
        hazards = np.column_stack((r1 * stock[:, ii-1, 0],
                                   r2 * stock[:, ii-1, 0] * stock[:, ii-1, 1],
                                   r3 * stock[:, ii-1, 1]))

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_sum_hazards = 1. / np.sum(hazards, axis=1, keepdims=True)  # inf if all dead

            delta_t = random_state.exponential(inv_sum_hazards.ravel())
            times[:, ii] = times[:, ii-1] + delta_t

            # choose reaction according to their probabilities
            probs = hazards * inv_sum_hazards
            cumprobs = np.cumsum(probs[:, :-1], axis=1)
            x = random_state.uniform(size=(batch_size, 1))
            reaction = np.sum(x >= cumprobs, axis=1)

        # null reaction if both populations dead
        reaction = np.where(np.isinf(inv_sum_hazards.ravel()), 3, reaction)

        # update stock
        stock[:, ii, :] = stock[:, ii-1, :] + stoichiometry[reaction, :]

        # no point to continue if predators = 0
        times[:, ii] = np.where(stock[:, ii, 1] == 0, time_end, times[:, ii])

    stock = stock[:, :ii+1, :]
    times = times[:, :ii+1]

    times_out = np.linspace(0, time_end, n_obs)
    stock_out = np.empty((batch_size, n_obs, 2), dtype=np.int32)
    stock_out[:, 0, :] = stock[:, 0, :]

    # observations at even intervals
    for ii in range(1, n_obs):
        iy, ix = np.where(times >= times_out[ii])
        iy, iix = np.unique(iy, return_index=True)
        ix = ix[iix] - 1
        time_term = (times_out[ii] - times[iy, ix]) / (times[iy, ix+1] - times[iy, ix])
        stock_out[:, ii, 0] = (stock[iy, ix+1, 0] - stock[iy, ix, 0]) * time_term \
            + stock[iy, ix, 0] + random_state.normal(scale=sigma, size=batch_size)
        stock_out[:, ii, 1] = (stock[iy, ix+1, 1] - stock[iy, ix, 1]) * time_term \
            + stock[iy, ix, 1] + random_state.normal(scale=sigma, size=batch_size)

    if return_full:
        return (stock_out, times_out, stock, times)

    return stock_out


def get_model(n_obs=50, true_params=None, seed_obs=None, **kwargs):
    """Return a complete Lotka-Volterra model in inference task.

    Parameters
    ----------
    n_obs : int, optional
        Number of observations.
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.

    Returns
    -------
    m : elfi.ElfiModel

    """
    logger = logging.getLogger()
    if true_params is None:
        true_params = [1.0, 0.005, 0.6, 50, 100, 10.]

    kwargs['n_obs'] = n_obs
    y_obs = lotka_volterra(*true_params, random_state=np.random.RandomState(seed_obs), **kwargs)

    m = elfi.ElfiModel()
    sim_fn = partial(lotka_volterra, **kwargs)
    priors = []
    sumstats = []

    priors.append(elfi.Prior(ExpUniform, -2, 0, model=m, name='r1'))
    priors.append(elfi.Prior(ExpUniform, -5, -2.5, model=m, name='r2'))  # easily kills populations
    priors.append(elfi.Prior(ExpUniform, -2, 0, model=m, name='r3'))
    priors.append(elfi.Prior('poisson', 50, model=m, name='prey0'))
    priors.append(elfi.Prior('poisson', 100, model=m, name='predator0'))
    priors.append(elfi.Prior(ExpUniform, np.log(0.5), np.log(50), model=m, name='sigma'))

    elfi.Simulator(sim_fn, *priors, observed=y_obs, name='LV')
    sumstats.append(elfi.Summary(partial(pick_stock, species=0), m['LV'], name='prey'))
    sumstats.append(elfi.Summary(partial(pick_stock, species=1), m['LV'], name='predator'))
    elfi.Distance('sqeuclidean', *sumstats, name='d')

    logger.info("Generated %i observations with true parameters r1: %.1f, r2: %.3f, r3: %.1f, "
                "prey0: %i, predator0: %i, sigma: %.1f.", n_obs, *true_params)

    return m


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
        u = ss.uniform.rvs(loc=a, scale=b-a, size=size, random_state=random_state)
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


def pick_stock(stock, species):
    """Return the stock for single species.

    Parameters
    ----------
    stock : np.array
    species : int
        0 for prey, 1 for predator.

    Returns
    -------
    np.array

    """
    return stock[:, :, species]
