import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

import elfi
from elfi.examples import ma2
from elfi.methods.bsl.select_penalty import select_penalty
# from elfi.methods.utils import logpdf

import time

np.random.seed(12345)

tic = time.time()

def MA2(x, n_obs=50, batch_size=1, random_state=None):
    #Make inputs 2d arrays for numpy  broadcasting with w
    # print('runningsim', x)
    t1 = x[0]
    t2 = x[1]
    # print('t1', t1, 't2', t2)
    # print(1/0)
    t1 = np.asanyarray(t1).reshape((-1, 1))
    t2 = np.asanyarray(t2).reshape((-1, 1))
    random_state = random_state or np.random
    w = random_state.randn(batch_size, n_obs+2) #i.i.d. sequence ~ N(0,1)
    x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
    return x

def autocov(x, lag=1):
    C = np.mean(x[:, lag:] * x[:, :-lag], axis=1)
    return C

m = elfi.new_model()

#true params
t1_true = 0.6
t2_true = 0.2

y_obs = MA2([t1_true, t2_true], n_obs=50)
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

# t1_1000 = CustomPrior_t1.rvs(2, 1000)
# t2_1000 = CustomPrior_t2.rvs(t1_1000, 1, 1000)
# plt.scatter(t1_1000, t2_1000, s=4, edgecolor='none');

# def ma2_prior(theta):
#     print('sum', sum(theta))
#     return (theta[0] > -2 and theta[0] < 2 and sum(theta) > -1 and (theta[1] - theta[2] < 1))

# elfi.Prior(ma2_prior)

Y = elfi.Simulator(MA2, observed=y_obs)

def dummy_func(x):
    # rows, col = x.shape[0], x.shape[1]
    # mean_col = np.mean(x, axis=1)
    # mean_col = mean_col.reshape(rows, 1)
    return x

S1 = elfi.Summary(dummy_func, Y)

x_obs = MA2([t1_true, t2_true])
# y_obs = autocov(x_obs)
print('x_obs', x_obs)
# print(1/0)

lmda_all = np.exp(np.arange(-5.5, -1.5, 0.2))

# print('lmda_all', lmda_all)
# select_penalty(x_obs, n=300, lmdas=lmda_all, M=10, theta=[t1_true, t2_true],
#                sigma=1.5, method="bsl", shrinkage="glasso",
#                sim_fn=MA2, sum_fn=dummy_func)

res = elfi.BSL(m['_summary'], batch_size=50, y_obs=x_obs,
               n_sims=500, method="bsl", shrinkage="glasso",
               penalty=0.0179).sample(200000,
               params0=np.array([t1_true, t2_true]),
               sigma_proposals=np.array([[0.01, 0.005], [0.005, 0.01]]) # half cov
               )

# print('S1', S1)
# S2 = elfi.Summary(autocov, Y, 2, name="sum2") #the optional keyword lag given val=2

# # Finish the model with the final node that calcs squared distance
# d = elfi.Distance('euclidean', S1)
# rej = elfi.Rejection(d, batch_size=10000)
# results = rej.sample(1000, n_sim=1000000)
# results.plot_marginals()
# plt.show()
# results.plot_pairs()
# plt.show()
# logitTransformBound =  np.array([[-1, 1],
                                #  [-1, 1]
                                # ])

# covariance_mat = np.array([[0.00428866 0.00125061] [0.00125061 0.16603027]]
# covariance_mat [[0.00444413 0.00143872][[0.00143872 0.00447685]]
# # pool = elfi.OutputPool(['t1', 't2'])
# res = elfi.BSL(m['_summary'], batch_size=50, y_obs=x_obs,
#                n_sims=500, method="uBsl").sample(2000,
#                params0=np.array([t1_true, t2_true]),
#                sigma_proposals=np.array([[0.01, 0.005], [0.005, 0.01]]) # half cov
#                )
# print(res)
# toc = time.time()

# np.save('t1_data_290121semi.npy', res.samples['t1'])
# np.save('t2_data_290121semi.npy', res.samples['t2'])

# print('totalruntime:', toc - tic)

# res.plot_marginals(selector=None, bins=None, axes=None)
# plt.show()
# res.plot_pairs()
# plt.show()
# # res.plot_traces()
# # plt.show()
# # ma2_model = ma2.get_model()
# # elfi.draw(ma2_model)
# # elfi.BSL(ma2_model['d'], batch_size=10000).sample(1000)
# print('ttttt')

