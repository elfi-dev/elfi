"""Example implementation of the Machand toad model.
This model simulates the movement of Fowler's toad species.
The model is proposed in MARCHAND.
TODO: REFERENCE HERE
"""

from functools import partial

import numpy as np
import scipy.stats as ss
import multiprocessing as mp
import time
import math

import elfi


# def rstable(alpha, gamma, n):  # logic for alpha-stable...
#     """ TODO: CMS ALGORITHM """
#     if alpha == 1:
#         print("TODO")
#         return
#     if alpha == 2:
#         print("TODO")
#         return
#     u = math.pi * (np.random.uniform(0, 1, n) - 0.5)
#     v = np.random.exponential(scale=1, size=n)
#     t = np.sin(alpha * u)/(np.cos(u) ** (1/alpha))  # TODO? check sinh or sin
#     s = (np.cos((1 - alpha) * u)/v)**((1 - alpha)/alpha)
#     return gamma * t * s


def toad(theta,
         n_toads=66,
         n_days=63,
         model=1,
         batch_size=1,
         random_state=None):
    """  # TODO """
    alpha, gamma, p0, random_state = [*theta]

    X = np.zeros((n_days, n_toads))
    random_state = random_state or np.random

    # TODO: MODEL 1 ONLY !
    for i in range(1, n_days):
        # i += 1
        if (model == 1): #  random return
            ind = random_state.uniform(0, 1, n_toads) >= p0
            non_ind = np.invert(ind)
            scipy_randomGen = ss.levy_stable
            scipy_randomGen.random_state = random_state
            delta_x = scipy_randomGen.rvs(alpha, beta=0, scale=gamma, size=np.sum(ind))
            # delta_x = rstable(alpha, gamma, np.sum(ind))  # logic levy_stable
            # delta_x = delta_x.reshape(-1, 1)
            X[i, ind] = X[i-1, ind] + delta_x
            non_ind_idx = np.argwhere(non_ind).flatten()

            ind_refuge = random_state.choice(i, size=len(non_ind_idx))
            X[i, non_ind_idx] = X[ind_refuge, non_ind_idx]

    return X


def toad_batch(alpha,
         gamma,
         p0,
         n_toads=66,
         n_days=63,
         model=1,
         batch_size=1,
         random_state=None):
    """  # TODO """
    # alpha, gamma, p0 = [*theta]
    if hasattr(alpha, '__len__') and len(alpha) > 1:
        pass
    else:  # assumes something array like passed in atm
        alpha = np.array([alpha])
        gamma = np.array([gamma])
        p0 = np.array([p0])

    X = np.zeros((n_days, n_toads, batch_size))
    random_state = random_state or np.random
    # TODO: MODEL 1 ONLY !
    for i in range(1, n_days):
        for j in range(batch_size):
            if (model == 1):  #  random return
                ind = random_state.uniform(0, 1, n_toads) >= p0[0]
                non_ind = np.invert(ind)

                scipy_randomGen = ss.levy_stable
                scipy_randomGen.random_state = random_state
                delta_x = scipy_randomGen.rvs(alpha[0], beta=0, scale=gamma[0], size=np.sum(ind))
                # test = rstable(alpha, gamma, np.sum(ind))
                # delta_x = delta_x.reshape(-1, 1)
                X[i, ind, j] = X[i-1, ind, j] + delta_x
                non_ind_idx = np.argwhere(non_ind).flatten()

                ind_refuge = random_state.choice(i, size=len(non_ind_idx))
                X[i, non_ind_idx, j] = X[ind_refuge, non_ind_idx, j]

    return X


# TODO: BETTER PARALLELISATION APPROACH
def sim_fun_wrapper(alpha, gamma, p0, n_toads=66, n_days=63, batch_size=1, random_state=None):
    if hasattr(alpha, '__len__') and len(alpha) > 1:
        # seeds = np.arange(0, len(alpha))
        N = len(alpha)
        # random_state = np.repeat(random_state, N)
        np_rand_ints = random_state.choice(N*10000, N, replace=False)
        random_states = [np.random.RandomState(rand_choice) for rand_choice in np_rand_ints]
        theta = np.array(list(zip(alpha, gamma, p0, random_states)))
    else:  # assumes something array like passed in atm
        theta = np.array([alpha, gamma, p0, random_state])

    # TODO: something with random state?
    model = 1
    sim_np = np.zeros((n_days, n_toads, batch_size))

    # theta = np.array([alpha, gamma, p0])
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    tic = time.time()
    if batch_size > 1:
        cpu_num = 4
        pool = mp.Pool(cpu_num)
        res = pool.map(toad, theta)
        pool.close()    

    else:
        res = toad(theta)
    np.save('toad_res.npy', res)
    toc = time.time()
    print('time', toc - tic)
    sim_np = res

    sim_np = np.array(sim_np)
    tmp_np = np.zeros((n_days, n_toads, batch_size))  # TODO: inefficient...
    if batch_size > 1:
        for i in range(batch_size):
            tmp_np[:, :, i] = sim_np[i, :, :]  # TODO: DO EFFICIENTLY?
    else:
        tmp_np = sim_np

    return tmp_np


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
    n_lag = len(lag)
    n_sims = 1
    if X.ndim == 3:
        x1, x2, n_sims = X.shape  # TODO: magic
    else:
        X = X.reshape(X.shape[0], X.shape[1], -1)
    ssx_all = np.empty((n_sims, 48))  # TODO: MAGIC
    for sim in range(n_sims):
        ssx = []
        for k in range(n_lag):
            X_sim = X[:, :, sim]
            disp = obs_mat_to_deltax(X_sim, lag[k])  # TODO: needed
            indret = np.array([disp[np.abs(disp) < 10]])
            noret = np.array([disp[np.abs(disp) > 10]])
            if noret.size == 0:  # safety check
                noret = np.array(np.inf)
                logdiff = np.repeat(np.inf, len(p)-1)
            else:
                logdiff = np.array([np.log(np.diff(np.quantile(np.abs(noret), p)))])
            ssx = np.concatenate((ssx, [indret.size], [np.median(np.abs(noret))], logdiff.flatten()))  # TODO?: check, changed to number abs less 10
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


def get_model(n_obs=None, true_params=None, seed_obs=None, parallel=True):
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

    if parallel:
        sim_fn = sim_fun_wrapper
        sim_fn = partial(sim_fn, n_toads=66, n_days=63)
    else:
        sim_fn = toad_batch

    y = sim_fn(*true_params, n_toads=66, n_days=63,
               random_state=np.random.RandomState(seed_obs))

    elfi.Prior('uniform', 1, 1, model=m, name='alpha')
    elfi.Prior('uniform', 0, 100, model=m, name='gamma')
    elfi.Prior('uniform', 0, 0.9, model=m, name='p0')
    elfi.Simulator(sim_fn, m['alpha'], m['gamma'], m['p0'], observed=y, 
                   name='toad')
    sum_stats = elfi.Summary(compute_summaries, m['toad'], name='S')
    elfi.Distance('euclidean', sum_stats, name='d')

    return m
