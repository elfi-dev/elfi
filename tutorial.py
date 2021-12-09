import time

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import logging
import elfi
from elfi.examples import ma2

seed = 1234
np.random.seed(1234)

# def MA2(t1, t2, n_obs=50, batch_size=1, random_state=None):
#     #Make inputs 2d arrays for numpy  broadcasting with w
#     t1 = np.asanyarray(t1).reshape((-1, 1))
#     t2 = np.asanyarray(t1).reshape((-1, 1))
#     random_state = random_state or np.random
#     w = random_state.randn(batch_size, n_obs+2) #i.i.d. sequence ~ N(0,1)

#     print('t1', t1.shape, 't2', t2.shape, 'w', w.shape)
#     x = w[:, 2:] + t1*w[:, 1:-1] + t2*w[:, :-2]
#     return x

# def autocov(x, lag=1):
#     C = np.mean(x[:, lag:] * x[:, :-lag], axis=1)
#     return C

# t1_true = 0.6
# t2_true = 0.2

# y_obs = MA2(t1_true, t2_true, n_obs=50)

# # print('ravel', y_obs, y_obs.ravel())

# # # Plot the observed sequence
# # plt.figure(figsize=(11, 6));
# # plt.plot(y_obs.ravel());

# # # To illustrate the stoachasticity, plot couple more obs w/ true param
# # plt.plot(MA2(t1_true, t2_true).ravel())
# # plt.plot(MA2(t1_true, t2_true).ravel())

# # plt.show()

# t1 = elfi.Prior(ss.uniform, 0, 2)
# t2 = elfi.Prior('uniform', 0, 2)

# Y = elfi.Simulator(MA2, t1, t2, observed=y_obs)

# S1 = elfi.Summary(autocov, Y)
# # S2 = elfi.Summary(autocov, Y, 2) #the optional keyword lag given val=2

# # Finish the model with the final node that calcs squared distance
# d = elfi.Distance('euclidean', S1)

# elfi.draw(d)
# plt.show()
# class CustomPrior_t1(elfi.Distribution):
#     def rvs(b, size=1, random_state=None):
#         u = ss.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
#         t1 = np.where(u < 0.5, np.sqrt(2.*u)*b - b, -np.sqrt(2.*(1.-u))*b +b )
#         return t1

# class CustomPrior_t2(elfi.Distribution):
#     def rvs(t1, a, size=1, random_state=None):
#         locs = np.maximum(-a-t1, t1-a)
#         scales = a - locs
#         t2 = ss.uniform.rvs(loc=locs, scale=scales, size=size, random_state=random_state)
#         return t2

# t1_1000 = CustomPrior_t1.rvs(2, 1000)
# t2_1000 = CustomPrior_t2.rvs(t1_1000, 1, 1000)
# # plt.scatter(t1_1000, t2_1000, s=4, edgecolor='none')
# # plt.show()

# t1.become(elfi.Prior(CustomPrior_t1, 2))
# t2.become(elfi.Prior(CustomPrior_t2, t1, 1))

# elfi.draw(d)

m = ma2.get_model()

dist = m['d'].generate(10)
print('S1', dist)
rej = elfi.Rejection(m['d'], batch_size=1000, seed=seed)

N = 10000
vis = dict(xlim=[-2, 2], ylim=[-1, 1])
result = rej.sample(N, quantile=0.01, vis=vis)

result.samples['t1'].mean()

result.summary()

result2 = rej.sample(N, threshold=0.2)

print(result2)


# Request for 1M sims.
rej.set_objective(1000, n_sim=1000000)

time0 = time.time()
time1 = time0 + 1
while not rej.finished and time.time() < time1:
    rej.iterate()

print(rej.extract_result())
print(rej.finished)

pool = elfi.ArrayPool(['t1', 't2', 'S1'])  #, 'S2'])
rej = elfi.Rejection(m['d'], batch_size=10000, pool=pool)
result3 = rej.sample(N, n_sim=1000000)
print(result3)

# d.become(elfi.Distance('cityblock', S1, S2, p=1))
rej = elfi.Rejection(m['d'], batch_size=10000, pool=pool)
result4 = rej.sample(N, n_sim=1000000)
print(result4)

result5 = rej.sample(N, n_sim=1200000)
print(result5)

arraypool = elfi.ArrayPool(['t1', 't2', 'Y', 'd'])
rej = elfi.Rejection(m['d'], batch_size=10000, pool=arraypool)
result6 = rej.sample(100, threshold=0.03)
print(result6)

arraypool.flush()

import os
print('FIles in', arraypool.path, 'are', os.listdir(arraypool.path))

np.load(arraypool.path + '/t1.npy')

arraypool.close()
name = arraypool.name
print(name)

arraypool = elfi.ArrayPool.open(name)
print('This pool has', len(arraypool), 'batches')

arraypool.delete()

#Verify the deletion
try:
    os.listdir(arraypool.path)
except FileNotFoundError:
    print("The directory is removed")

# result5.plot_marginals()
# plt.show()
# result5.plot_pairs()
# plt.show()

class CustomPrior_t1(elfi.Distribution):
    def rvs(b, size=1, random_state=None):
        u = ss.uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
        t1 = np.where(u<0.5, np.sqrt(2.*u)*b - b, -np.sqrt(2.*(1.-u))*b + b)
        return t1
    def pdf(x, b):
        p = 1./b - np.abs(x) / (b*b)
        p = np.where(p < 0., 0., p) #disallow values outside of [-b, b]
        return p

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
        p = np.where(scales>0., p, 0.) #disallow values outside of [-b, b]
        return p

#redefine the priors
t1.become(elfi.Prior(CustomPrior_t1, 2, model=t1.model))
t2.become(elfi.Prior(CustomPrior_t2, t1, 1))

smc = elfi.SMC(d, batch_size=10000, seed=seed)
N = 1000
schedule = [0.7, 0.2, 0.05]
result_smc = smc.sample(N, schedule)
result_smc.summary(all=True)

result_smc.sample_means_summary(all=True)

result_smc.plot_marginals(all=True, bins=25, figsize=(8, 2), fontsize=12)

n_populations = len(schedule)
fig, ax = plt.subplots(ncols=n_populations, sharex=True, sharey=True, figsize=(16, 6))

for i, pop in enumerate(result_smc.populations):
    s = pop.samples
    ax[i].scatter(s['t1'], s['t2'], s=5, edgecolor='none')
    ax[i].set_title("Population {}".format(i))
    ax[i].plot([0, 2, -2, 0], [-1, 1, 1, -1], 'b')
    ax[i].set_xlabel('t1');

ax[0].set_ylabel('t2')
ax[0].set_xlim([-2, 2])
ax[0].set_ylim([-1, 1])

plt.show()
