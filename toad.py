import numpy as np
import math
# import matlab.engine
import elfi
import matplotlib.pyplot as plt
import multiprocessing as mp
from simulate_toads2 import simulate_toads2
import sys
import time
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
from elfi.methods.bsl.select_penalty import select_penalty



#     import cProfile
#     # if check avoids hackery when not profiling
#     # Optional; hackery *seems* to work fine even when not profiling, it's just wasteful
#     if sys.modules['__main__'].__file__ == cProfile.__file__:
#         import toad  # Imports you again (does *not* use cache or execute as __main__)
#         globals().update(vars(toad))  # Replaces current contents with newly imported stuff
#         sys.modules['__main__'] = toad  # Ensures pickle lookups on __main__ find matching version
#     # main()  # Or series of statements

# eng = matlab.engine.start_matlab()

alpha = 1.8
delta = 35
p_0 = 0.6

theta = [alpha, delta, p_0]
n_toads = 66
n_days = 63
model_num = 1

# theta = matlab.double(theta)

# eng = matlab.engine.start_matlab()
X_obs = simulate_toads2(*theta, n_toads, n_days, model_num) # 1 - nobs, batch_size
# eng.quit()

m = elfi.new_model()

t1 = elfi.Prior('uniform', 1, 1)
t2 = elfi.Prior('uniform', 0, 100)
t3 = elfi.Prior('uniform', 0, 0.9)

def run_simulation(theta, ntoads=66, ndays=63, model=1):
    # print('theta', theta)
    alpha = theta[0]
    delta = theta[1]
    p0 = theta[2]
    sim_res = simulate_toads2(alpha, delta, p0, ntoads, ndays, model)
    return sim_res

def compute_summary(X, lag=[1,2,4,8], p=np.linspace(0,1,11)):
    ssx = []
    n_lag = len(lag)
    for k in range(n_lag):
        # tic = time.time()
        disp = obs_mat_to_deltax(X, lag[k])
        # toc = time.time()
        # print('obs_mat_calc_time', toc - tic)
        indret = np.array([disp[disp < 10]])
        noret = np.array([disp[disp > 10]])
        logdiff = np.array([np.log(np.diff(np.quantile(noret, p)))])
        ssx = np.concatenate((ssx, [np.mean(indret)], [np.median(noret)], logdiff.flatten()))
    return ssx


def sim_fun_wrapper(alpha, delta, p0, n_obs, batch_size=None):
    # NOTE: batch_size not actually doing anything right now
    ntoads = 66
    ndays = 63
    model = 1
    sim_np = np.zeros((ndays, ntoads, n_obs))
    print('n_obs', n_obs, 'batch_size', n_obs)
    print('inthewrapper')
    tic = time.time()
    pool = mp.Pool(mp.cpu_count())
    print('alpha, delta, p0', alpha, delta, p0)
    theta = np.array([alpha, delta, p0])
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    sim_np = pool.map(run_simulation, [theta]*n_obs)
    # for i in range(batch_size):
    #     sim_np[:, :, i] = run_simulation(x)
    pool.close()
    toc = time.time()
    print('timetakenforsims', toc - tic)
    sim_np = np.array(sim_np)
    sim_np = sim_np.reshape((ntoads, ndays, n_obs))
    return sim_np

def obs_mat_to_deltax(X, lag):
    n_days = X.shape[0]
    n_toads = X.shape[1]
    x = np.zeros(n_toads * (n_days - lag))
    for i in range(n_days - lag):
        j = i + lag
        x0 = X[i, j]
        x1 = X[i+lag, j]
        deltax = X[j, :] - X[i, :]
        deltax = deltax[np.logical_not(np.isnan(deltax))]
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

def toad_sum(X, lag=[1,2,4,8], p=np.linspace(0,1,11)):
    n_lag = len(lag)
    # ssx = np.zeros()
    # if X == 3: #  to avoid error for initial summary stat
    X = np.asarray(X)
    X = X.reshape((63, 66, -1))
    if X.ndim == 3:
        num_sims = X.shape[2]
    num_summary_stats = 48 # random number
    ssx_mat = np.zeros((num_sims, num_summary_stats))
    pool = mp.Pool(mp.cpu_count())
    # tic = time.time()
    # sum_np = np.zeros((48, num_sims))
    # for i in range(num_sims):
        # sum_np[:, i] = compute_summary(X[:, :, i])
    sum_np = pool.map(compute_summary, [X[:, :, i] for i in range(num_sims)])
    # toc = time.time()
    pool.close()
    # print('time for one toad sum', (toc - tic))
    ssx_mat = np.array(sum_np)
    ssx_mat = ssx_mat.reshape((-1, 48)) # TODO: MAGIC !
    print('ssx_mat', ssx_mat.shape)
    return ssx_mat

Y = elfi.Simulator(sim_fun_wrapper, t1, t2, t3, observed=X_obs)

y_obs_sum = toad_sum(X_obs)
print('y_obs_sum', y_obs_sum)
elfi.Summary(toad_sum, Y)
# print(1/0)

logitTransformBound = np.array([[1, 2],
                                [0, 100],
                                [0, 0.9]
                                ])

# est_approx_posterior_cov = np.array([[1.52033025e-03, 2.58101839e-02, 1.53252942e-03],
#  [2.58101839e-02, 6.12685886e+00, 3.90823700e-02],
#  [1.53252942e-03, 3.90823700e-02, 2.00253133e-03]])

cov = np.array([[0.081, 0.007, 0.001],
  [0.007, 0.003, 0.001],
  [0.001, 0.001, 0.003]])

# # uncomment for whitening matrix
# sim_mat = sim_fun_wrapper(alpha, delta, p_0, 20000)
# sum_mat = toad_sum(sim_mat)
# W = estimate_whitening_matrix(sum_mat)

# np.save("est_whitening_mat.npy", W)
W = np.load("est_whitening_mat.npy")
# lmdas = list(np.arange(0.2, 0.9, 0.01))
# penalty = select_penalty(ssy=y_obs_sum.flatten(), n=100, lmdas=lmdas, M=20, theta=theta, shrinkage="warton",
#                         sim_fn=sim_fun_wrapper, sum_fn=toad_sum, n_obs=100, 
#                         whitening=W)

# another cov matrix
# [[ 4.31073146e-03 -1.67239272e-01  3.43900135e-03]
#  [-1.67239272e-01  1.32992663e+01 -9.25558899e-02]
#  [ 3.43900135e-03 -9.25558899e-02  3.50688622e-03]]


# est_approx_posterior_cov = np.eye(3)
tic = time.time()
res = elfi.BSL(m['_summary'], batch_size=44, y_obs=y_obs_sum, n_sims=44,
              logitTransformBound=logitTransformBound, method="bsl", n_obs=44,
              penalty=0, shrinkage="warton", whitening=W,
              ).sample(2000,
               params0=np.array(theta), sigma_proposals=0.1*cov)
# print(np.cov(res))
# res.plot_marginals(selector=None, bins=None, axes=None)
toc = time.time()

print('runtime:', toc - tic)

np.save('t1_data_280121toad.npy', res.samples['t1'])
np.save('t2_data_280121toad.npy', res.samples['t2'])
np.save('t3_data_280121toad.npy', res.samples['t2'])


res.plot_pairs()
plt.show()

# if __name__ == '__main__':
#     exit(cProfile.run('main()'))
