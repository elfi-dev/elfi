"""Example implementation of the Machand toad model.
TODO: REFERENCE HERE
"""

from functools import partial

import numpy as np
import scipy.stats as ss
import multiprocessing as mp
import time
import math
# import pandas as pd  # TODO: REMOVE MODULE - DEBUGGING

import elfi


def rstable(alpha, gamma, n):
    """ TODO: CMS ALGORITHM """
    if alpha == 1:
        print("TODO")
        return
    if alpha == 2:
        print("TODO")
        return
    u = math.pi * (np.random.uniform(0, 1, n) - 0.5)
    v = np.random.exponential(scale=1, size=n)
    t = np.sinh(alpha * u)/(np.cos(u) ** (1/alpha))
    s = (np.cos((1 - alpha) * u)/v)**((1 - alpha)/alpha)
    return gamma * t * s


def toad(theta,
         n_toads=66,
         n_days=63,
         model=1,
         batch_size=1,
         random_state=None):
    """  # TODO """
    alpha, gamma, p0 = [*theta]
    X = np.zeros((n_days, n_toads)) #, batch_size))
    # TODO: MODEL 1 ONLY !
    for i in range(1, n_days):
        # i += 1
        if (model == 1): #  random return
            ind = np.random.uniform(0, 1, n_toads) >= p0
            non_ind = np.invert(ind)

            delta_x = ss.levy_stable.rvs(alpha, beta=0, scale=gamma, size=np.sum(ind))
            # test = rstable(alpha, gamma, np.sum(ind))
            # delta_x = delta_x.reshape(-1, 1)
            X[i, ind] = X[i-1, ind] + delta_x
            non_ind_idx = np.argwhere(non_ind).flatten()

            ind_refuge = np.random.choice(i, size=len(non_ind_idx))
            # idx = np.ravel_multi_index((ind_refuge, np.argwhere(non_ind)), X.shape)
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
    # TODO: MODEL 1 ONLY !
    for i in range(1, n_days):
        for j in range(batch_size):
            # i += 1
            if (model == 1): #  random return
                ind = np.random.uniform(0, 1, n_toads)>= p0[0]
                non_ind = np.invert(ind)

                delta_x = ss.levy_stable.rvs(alpha[0], beta=0, scale=gamma[0], size=np.sum(ind))
                # test = rstable(alpha, gamma, np.sum(ind))
                # delta_x = delta_x.reshape(-1, 1)
                X[i, ind, j] = X[i-1, ind, j] + delta_x
                non_ind_idx = np.argwhere(non_ind).flatten()

                ind_refuge = np.random.choice(i, size=len(non_ind_idx))
                # idx = np.ravel_multi_index((ind_refuge, np.argwhere(non_ind)), X.shape)
                X[i, non_ind_idx, j] = X[ind_refuge, non_ind_idx, j]

    return X


# TODO: BETTER PARALLELISATION APPROACH
def sim_fun_wrapper(alpha, gamma, p0, n_toads=66, n_days=63, batch_size=1, random_state=None):
    if hasattr(alpha, '__len__') and len(alpha) > 1:
        theta = np.array(list(zip(alpha, gamma, p0)))
    else:  # assumes something array like passed in atm
        theta = np.array([alpha, gamma, p0])

    # TODO: something with random state?
    model = 1
    sim_np = np.zeros((n_days, n_toads, batch_size))

    # theta = np.array([alpha, gamma, p0])
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    tic = time.time()
    if batch_size > 1:
        # pool = mp.Pool(1)
        print('mp.cpu_count()', mp.cpu_count())
        pool = mp.Pool(mp.cpu_count())
        res = pool.map(toad, theta)
        pool.close()
    else:
        res = toad(theta)
    toc = time.time()
    print('time', toc - tic)
    sim_np = res
    # for i in range(batch_size):
    #     sim_np[:, :, i] = run_simulation(x)

    sim_np = np.array(sim_np)
    tmp_np = np.zeros((n_days, n_toads, batch_size))  # TODO: inefficient...
    if batch_size > 1:
        for i in range(batch_size):
            tmp_np[:, :, i] = sim_np[i, :, :]
    else:
        tmp_np = sim_np
    # sim_np = sim_np.reshape((n_days, n_toads, batch_size))  # TODO: check shapes

    return tmp_np


def compute_summaries(X, lag=[1, 2, 4, 8], p=np.linspace(0, 1, 11)):
    """ # TODO """
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
            # tic = time.time()
            disp = obs_mat_to_deltax(X_sim, lag[k])  # TODO: needed
            # disp = np.abs(disp)  #TODO ; fix rest abs
            # toc = time.time()
            # print('obs_mat_calc_time', toc - tic)
            indret = np.array([disp[np.abs(disp) < 10]])
            noret = np.array([disp[np.abs(disp) > 10]])
            if noret.size == 0:  # safety check
                noret = np.array(np.inf)
                logdiff = np.repeat(np.inf, len(p)-1)
                # noret = np.array([np.inf] * len(indret))
            else:
                logdiff = np.array([np.log(np.diff(np.quantile(np.abs(noret), p)))])
            ssx = np.concatenate((ssx, [indret.size], [np.median(np.abs(noret))], logdiff.flatten()))  # TODO?: check, changed to number abs less 10
        ssx_all[sim, :] = ssx
    return ssx_all


def obs_mat_to_deltax(X, lag):
    n_days = X.shape[0]
    n_toads = X.shape[1]
    x = np.zeros(n_toads * (n_days - lag))
    for i in range(n_days - lag):
        j = i + lag
        # x0 = X[i, j]
        # x1 = X[i+lag, j]
        deltax = X[j, :] - X[i, :]
        # deltax = deltax[np.logical_not(np.isnan(deltax))]
        x[i*n_toads:(i*n_toads+n_toads)] = deltax
    # for j in range(n_toads):
    #     for i in range(n_days - lag):
    #         x0 = X[i, j]
    #         x1 = X[i+lag, j]
    #         if (np.isnan(x0) or np.isnan(x1)):
    #             print('isnan')
    #             continue
    #         temp = x1 - x0
    #         x = np.append(x, np.abs(temp))
    # print('x', x)
    return x


def get_model(n_obs=None, true_params=None, seed_obs=None, parallel=True):
    """ # TODO: """
    if true_params is None:
        true_params = [1.7, 35.0, 0.6]

    m = elfi.ElfiModel()

    if parallel:
        sim_fn = sim_fun_wrapper
        sim_fn = partial(sim_fn, n_toads=66, n_days=63)
    else:
        sim_fn = toad_batch

    y = sim_fn(*true_params, n_toads=66, n_days=63, random_state=np.random.RandomState(seed_obs))
    # y
    # print('yyyy', y[2, 2, :])
    # print('y1', y.shape)
    ssy = compute_summaries(y)
    # df_y = pd.DataFrame(np.atleast_2d(ssy))
    # df_y.to_csv("y_obs.csv")
    # y = pd.read_csv("y_obs.csv")
    # y = y.to_numpy()
    # print('y', y.shape)

    elfi.Prior('uniform', 1, 1, model=m, name='alpha')
    elfi.Prior('uniform', 0, 100, model=m, name='gamma')
    elfi.Prior('uniform', 0, 0.9, model=m, name='p0')
    elfi.Simulator(sim_fn, m['alpha'], m['gamma'], m['p0'], observed=y, name='toad')
    sum_stats = elfi.Summary(compute_summaries, m['toad'], name='S')
    elfi.Distance('euclidean', sum_stats, name='d')

    return m


