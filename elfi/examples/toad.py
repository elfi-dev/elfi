"""Example implementation of the Machand toad model.
This model simulates the movement of Fowler's toad species.
"""

from functools import partial

import numpy as np
import scipy.stats as ss
import multiprocessing as mp
import time
import math

import elfi


def toad(alpha,
         gamma,
         p0,
         random_state=None,
         n_toads=66,
         n_days=63,
         model=1,
         batch_size=1,
        #  random_state=None,
         *args,
         **kwargs):
    """Sample the movement of Fowler's toad species.

    Models foraging steps using a levy_stable distribution where individuals
    either return to a previous site or establish a new one.

    Parameters
    ----------
    theta : np.array
        Vector of proposed parameter values:
            alpha: stability parameter
            gamma: scale parameter
            p0: probability of returning to their previous refuge site.
        Seed... ensures same results including when parallelised.
    n_toads : int, optional
    n_days : int, optional
    model : int, optional
        1 = random return, a distance-independent probability of return
            to any previous refuge.
    batch_size : int, optional
    random_state : RandomState, optional
    References
    ----------
    Marchand, P., Boenke, M., and Green, D. M. (2017).
    A stochastic movement model reproduces patterns of site fidelity and long-
    distance dispersal in a population of fowlers toads (anaxyrus fowleri).
    Ecological Modelling,360:63–69.
    """
    # print('alpha', alpha)
    # print('gamma', gamma)
    # print('p0', p0)
    X = np.zeros((n_days, n_toads))
    # print('random_state', random_state)
    random_state = random_state or np.random

    for i in range(1, n_days):
        if (model == 1):  # random return
            ind = random_state.uniform(0, 1, n_toads) >= p0
            non_ind = np.invert(ind)
            scipy_randomGen = ss.levy_stable
            scipy_randomGen.random_state = random_state
            delta_x = scipy_randomGen.rvs(alpha, beta=0, scale=gamma,
                                          size=np.sum(ind))
            X[i, ind] = X[i-1, ind] + delta_x
            non_ind_idx = np.argwhere(non_ind).flatten()

            ind_refuge = random_state.choice(i, size=len(non_ind_idx))
            X[i, non_ind_idx] = X[ind_refuge, non_ind_idx]
    # print('randX', X[15, 16])
    return X


def toad_batch(alpha,
               gamma,
               p0,
               n_toads=66,
               n_days=63,
               model=1,
               batch_size=1,
               random_state=None):
    """See toad function for same details. This function is used when
    no parallelisation is used.
    """
    if hasattr(alpha, '__len__') and len(alpha) > 1:
        pass
    else:  # assumes something array like passed
        alpha = np.array([alpha])
        gamma = np.array([gamma])
        p0 = np.array([p0])

    X = np.zeros((n_days, n_toads, batch_size))
    random_state = random_state or np.random
    for i in range(1, n_days):
        for j in range(batch_size):
            if (model == 1):  # random return
                ind = random_state.uniform(0, 1, n_toads) >= p0[0]
                non_ind = np.invert(ind)

                scipy_randomGen = ss.levy_stable
                scipy_randomGen.random_state = random_state
                delta_x = scipy_randomGen.rvs(alpha[0], beta=0, scale=gamma[0],
                                              size=np.sum(ind))
                X[i, ind, j] = X[i-1, ind, j] + delta_x
                non_ind_idx = np.argwhere(non_ind).flatten()

                ind_refuge = random_state.choice(i, size=len(non_ind_idx))
                X[i, non_ind_idx, j] = X[ind_refuge, non_ind_idx, j]

    return X


def sim_fun_wrapper(alpha, gamma, p0, n_toads=66, n_days=63, batch_size=1,
                    random_state=None):
    """Function to parallelise toad function
    """
    # TODO! USE SeedSequence?
    if hasattr(alpha, '__len__') and len(alpha) > 1:
        N = len(alpha)
        np_rand_ints = random_state.choice(N*10000, N, replace=False)
        random_states = [np.random.RandomState(rand_choice) for rand_choice
                         in np_rand_ints]
        theta = np.array(list(zip(alpha, gamma, p0, random_states)))
    else:  # assumes something array like passed in atm
        theta = np.array([alpha, gamma, p0, random_state])

    model = 1
    sim_np = np.zeros((n_days, n_toads, batch_size))

    if batch_size > 1:
        cpu_num = 4  # change cpu num for parallelisation
        pool = mp.Pool(cpu_num)
        res = pool.map(toad, theta)
        pool.close()
    else:
        res = toad(theta)
    toc = time.time()
    res = reshape_res(res)

    return res


def reshape_res(res, batch_size=1):
    sim_np = res
    sim_np = np.array(sim_np)
    batch_size = sim_np.size // (63 * 66)  # as size=batch_size*ndays*ntoads
    tmp_np = np.zeros((63, 66, batch_size))
    if batch_size > 1:
        for i in range(batch_size):
            tmp_np[:, :, i] = sim_np[i, :, :]
    else:
        tmp_np = sim_np
    return tmp_np


# def test_parallelise(fn, seed=None):
    # ss = np.random.SeedSequence()


def compute_summaries(X, lag=[1, 2, 4, 8], p=np.linspace(0, 1, 11)):
    """ Function to compute all 48 summaries.
        For each lag...
            Log of the differences in the 0, 0.1, ..., 1 quantiles
            The number of absolute displacements less than 10m
            Median of the absolute displacements greater than 10m

        Parameters
        ----------
        X : np.array of shape (ndays x ntoads)
            observed matrix of toad displacements
        lag : list of ints, optional
            the number of days behind to compute displacement with
        p : np.array, optional

        Returns
        -------
        x : A vector of displacements
    """
    X = reshape_res(X)
    n_lag = len(lag)
    n_sims = 1
    n_summaries = 48
    if X.ndim == 3:
        x1, x2, n_sims = X.shape
    else:
        X = X.reshape(X.shape[0], X.shape[1], -1)
    ssx_all = np.empty((n_sims, n_summaries))
    for sim in range(n_sims):
        ssx = []
        for k in range(n_lag):
            X_sim = X[:, :, sim]
            disp = obs_mat_to_deltax(X_sim, lag[k])
            indret = np.array([disp[np.abs(disp) < 10]])
            noret = np.array([disp[np.abs(disp) > 10]])
            if noret.size == 0:  # safety check
                noret = np.array(np.inf)
                logdiff = np.repeat(np.inf, len(p)-1)
            else:
                logdiff = np.array([np.log(np.diff(np.quantile(np.abs(noret),
                                                   p)))])
            ssx = np.concatenate((ssx,
                                  [indret.size],
                                  [np.median(np.abs(noret))],
                                  logdiff.flatten()))
        ssx_all[sim, :] = ssx
    return ssx_all


def obs_mat_to_deltax(X, lag):
    """Converts an observation matrix to a vector of displacements

    Parameters
    ----------
    X : np.array (n_days x n_toads)
        observed matrix of toad displacements
    lag : int
        the number of days behind to compute displacement with

    Returns
    -------
    x : A vector of displacements
    """
    n_days = X.shape[0]
    n_toads = X.shape[1]
    x = np.zeros(n_toads * (n_days - lag))
    for i in range(n_days - lag):
        j = i + lag
        deltax = X[j, :] - X[i, :]
        x[i*n_toads:(i*n_toads+n_toads)] = deltax
    return x

# TODO? random testing... boxcox transformation for positive summaries?


def get_model(n_obs=None, true_params=None, seed_obs=None, parallelise=True,
              n_cpus=4):
    """Return a complete toad model in inference task.

    Parameters
    ----------
    n_obs : int, optional
        observation length of the MA2 process
    true_params : list, optional
        parameters with which the observed data is generated
    seed_obs : int, optional
        seed for the observed data generation
    parallel : bool, optional
        option to turn on or off parallel simulations
    Returns
    -------
    m : elfi.ElfiModel
    """
    if true_params is None:
        true_params = [1.7, 35.0, 0.6]

    m = elfi.ElfiModel()

    # if parallel:
    #     sim_fn = sim_fun_wrapper
    sim_fn = partial(toad, n_toads=66, n_days=63)
    # else:
    #     sim_fn = toad_batch

    y = toad(*true_params, random_state=np.random.RandomState(seed_obs))

    elfi.Prior('uniform', 1, 1, model=m, name='alpha')
    elfi.Prior('uniform', 0, 100, model=m, name='gamma')
    elfi.Prior('uniform', 0, 0.9, model=m, name='p0')
    elfi.Simulator(toad, m['alpha'], m['gamma'], m['p0'], observed=y,
                   name='toad', parallelise=parallelise, n_cpus=n_cpus)
    sum_stats = elfi.Summary(compute_summaries, m['toad'], name='S')
    # NOTE: toad written for BSL, distance node included but not tested
    elfi.Distance('euclidean', sum_stats, name='d')

    return m
