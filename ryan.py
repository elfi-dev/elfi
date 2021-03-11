import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

import elfi
from elfi.examples import ma2
import time
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
np.random.seed(12345)

tic = time.time()

def MA2(t1, t2, n_obs=100, batch_size=1, random_state=None):
    #Make inputs 2d arrays for numpy  broadcasting with w
    # print('runningsim', x)
    # print('t1', t1, 't2', t2)
    # print(1/0)
    t1 = np.asanyarray(t1).reshape((-1, 1))
    t2 = np.asanyarray(t2).reshape((-1, 1))
    random_state = random_state or np.random
    w = random_state.randn(batch_size, n_obs+2) #i.i.d. sequence ~ N(0,1)
    x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
    return x

# def autocov(x, lag=1):
#     C = np.mean(x[:, lag:] * x[:, :-lag], axis=1)
#     return C

m = elfi.new_model()

#true params
t1_true = 0.6
t2_true = 0.2
n_obs = 100 #TODO: Future params depend solely batch_size, n_bathes
y_obs = MA2(t1_true, t2_true, n_obs=n_obs)
y_obs = y_obs.flatten()

t1 = elfi.Prior(ss.uniform, 0, 2)
t2 = elfi.Prior('uniform', 0, 2)

class CustomPrior_t1(elfi.Distribution):
    def rvs(b, size=1, random_state=None):
        u = ss.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
        t1 = np.where(u<0.5, np.sqrt(2.*u)*b-b, -np.sqrt(2.*(1.-u))*b+b)
        return t1

    def pdf(x, b):
        p = 1./b - np.abs(x) / (b*b)
        p = np.where(p < 0., 0., p)  # disallow values outside of [-b, b] (affects weights only)
        return p


# define prior for t2 conditionally on t1 as in Marin et al., 2012, in range [-a, a]
class CustomPrior_t2(elfi.Distribution):
    def rvs(t1, a, size=1, random_state=None):
        locs = np.maximum(-a-t1, t1-a)
        scales = a - locs
        t2 = ss.uniform.rvs(loc=locs, scale=scales, size=size, random_state=random_state)
        return t2

    def pdf(x, t1, a):
        locs = np.maximum(-a-t1, t1-a)
        scales = a - locs
        p = ss.uniform.pdf(x, loc=locs, scale=scales)
        p = np.where(scales>0., p, 0.)  # disallow values outside of [-a, a] (affects weights only)
        return p


# Redefine the priors
t1.become(elfi.Prior(CustomPrior_t1, 2, model=t1.model))
t2.become(elfi.Prior(CustomPrior_t2, t1, 1))

# def ma2_prior(theta):
#     return (theta[0] > -2 and theta[0] < 2 and sum(theta) > -1 and (theta[1] - theta[2] < 1))

# elfi.Prior(ma2_prior)

Y = elfi.Simulator(MA2)

def dummy_func(x):
    # rows, col = x.shape[0], x.shape[1]
    # mean_col = np.mean(x, axis=1)
    # mean_col = mean_col.reshape(rows, 1)
    print('running the summary func')
    return np.transpose(x)

S1 = elfi.Summary(dummy_func, Y)
# S2 = elfi.Summary(autocov, Y, 2)
# d = elfi.Distance('euclidean', S1)
# x_obs = MA2(t1_true, t2_true)
# y_obs = autocov(x_obs)



elfi.draw(m)
plt.show()
# print(1/0)


# new_m = ma2.get_model()

# find whitening matrix
sim_mat = MA2(t1_true, t2_true, n_obs=n_obs, batch_size=200000)
W = estimate_whitening_matrix(sim_mat, method="semiBsl") # summary same as simulation here

lmdas = list(np.arange(0, 0.9, 0.01))
# penalty = select_penalty(y_obs, n=320, lmdas=lmdas, M=50, sigma=1.2,
#                          theta=[0.6, 0.2], shrinkage="warton",
#                          sim_fn= MA2, sum_fn=dummy_func, whitening=W)
# print('penalty', penalty)
# print(1/0)
# pool = elfi.OutputPool(['t1', 't2'])
res = elfi.BSL(m["_summary"], batch_size=320, n_batches=1, y_obs=y_obs,
               n_sims=320, method="bslmisspec", n_obs=n_obs,
               shrinkage="warton", penalty=0.14, whitening=W, type_misspec="mean"
               ).sample(2000,
               params0=np.array([t1_true, t2_true]),
               sigma_proposals=np.array([[0.1, 0.05], [0.05, 0.1]]) # half cov
               )
# res.infer(n_sim=2000)
print(res)
toc = time.time()

# np.save('t1_data_290121semi.npy', res.samples['t1'])
# np.save('t2_data_290121semi.npy', res.samples['t2'])

print('totalruntime:', toc - tic)

res.plot_marginals(selector=None, bins=None, axes=None)
plt.show()
res.plot_pairs()
plt.show()
# res.plot_traces()
# plt.show()
# ma2_model = ma2.get_model()
# elfi.draw(ma2_model)
# elfi.BSL(ma2_model['d'], batch_size=10000).sample(1000)
print('ttttt')

