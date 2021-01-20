import numpy as np
import math
import matlab.engine
import elfi

eng = matlab.engine.start_matlab()

alpha = 1.8
delta = 45
p_0 = 0.6

theta = [alpha, delta, p_0]
n_toads = 66
n_days = 63
model_num = 1

theta = matlab.double(theta)


X_obs = eng.simulate_toads2(theta, n_toads, n_days, model_num)

m = elfi.new_model()

t1 = elfi.Prior('uniform', 1, 1)
t2 = elfi.Prior('uniform', 0, 100)
t3 = elfi.Prior('uniform', 0, 0.9)

def sim_fun_wrapper(theta, batch_size):
    ntoads = 66
    ndays = 63
    model = 1
    sim_np = np.zeros((ndays, ntoads, batch_size))
    for sim_num in range(batch_size):
      theta_mat = matlab.double(theta.tolist())
      sim_res = eng.simulate_toads2(theta_mat, ntoads, ndays, model)
      sim_np[:, :, sim_num] = np.asarray(sim_res)
    return sim_np

def obs_mat_to_deltax(X, lag):
    n_days = X.shape[0]
    n_toads = X.shape[1]
    x = []
    for j in range(n_toads):
        for i in range(n_days - lag):
            x0 = X[i, j]
            x1 = X[i+lag, j]
            if (np.isnan(x0) or np.isnan(x1)):
                continue
            temp = x1 - x0
            x = np.append(x, np.abs(temp))
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
    for sim_num in range(num_sims):
        print('sim_num', sim_num)
        ssx = []
        for k in range(n_lag):
            disp = obs_mat_to_deltax(X[:, :, sim_num], lag[k])
            indret = np.array([disp[disp < 10]])
            # print('indret', indret.shape)
            # print('indretmean', np.mean(indret))
            noret = np.array([disp[disp > 10]])
            # print('noret', noret.shape)
            # print('noretmean', np.mean(noret))
            # noret = disp[!indret]
            # print('disp', disp)
            logdiff = np.array([np.log(np.diff(np.quantile(noret, p)))])
            # print('logdiff', logdiff, logdiff.shape)
            # print('types', type(indret), type(noret), type(logdiff))
            ssx = np.concatenate((ssx, [np.mean(indret)], [np.median(noret)], logdiff.flatten()))
            # print('indret', len(indret), 'noret', len(noret), 'logdiff', len(logdiff))
            # print('len(ssx)', len(ssx))
            # print('ssx', ssx)
            # print(1/0)
        ssx_mat[sim_num, :] = ssx
    print('ssx_mat', ssx_mat)
    return ssx_mat

Y = elfi.Simulator(sim_fun_wrapper, t1, t2, t3, observed=X_obs)

y_obs_sum = toad_sum(X_obs)
print('y_obs_sum', y_obs_sum)
elfi.Summary(toad_sum, Y)

logitTransformBound = np.array([[1, 2],
                                [0, 100],
                                [0, 0.9]
                                ])
res = elfi.BSL(m['_simulator'], batch_size=100, y_obs=y_obs_sum, n_sims=50,
              logitTransformBound=logitTransformBound).sample(1000,
               params0=np.array(theta))
# print(res)
# res.plot_marginals(selector=None, bins=None, axes=None)
# res.plot_pairs()
# plt.show()
eng.quit()
