from elfi.examples.gnk import get_model, ss_robust, ss_octile, GNK
import elfi
import numpy as np
import matplotlib.pyplot as plt
import time
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
true_params = [3, 1, 2, 0.5] # a, b, g, k
np.random.seed(123)

n_obs=1000

model = get_model(n_obs=n_obs, true_params=true_params)

# def sim_gnk(x):
    # GNK

sim = GNK(n_obs=n_obs, *true_params)
# print('sim', sim.shape)
# sum_stat1 = np.array([sim[i, 0, :] for i in range(1000)])
# sum_stat1 = sum_stat1.flatten()
# plt.hist(sum_stat1, density=True, bins=30)
# plt.show()
# print('sum_stat1', sum_stat1)
# print(1/0)
# sim = sim.flatten()
y_obs = ss_robust(sim)
y_obs = y_obs.flatten()
print('y_obs', y_obs)

tic = time.time()
# lmdas = list(np.exp(np.arange(-5.5, 1.5, 0.1)))

est_post_cov = np.array(
    [[ 0.00058258,  0.00044668, -0.0002838,  -0.00022737],
 [ 0.00044668,  0.00256855,  0.00292011, -0.00284012],
 [-0.0002838,   0.00292011,  0.01953341, -0.00134999],
 [-0.00022737, -0.00284012, -0.00134999,  0.00590356]])

sim_mat = np.zeros((1000, 4))  # TODO: Magic numbers...
for i in range(1000):
    sim_mat[i, :] = ss_robust(GNK(n_obs=n_obs, *true_params)).flatten() # obs as row


w_pca = estimate_whitening_matrix(sim_mat)
print('w_pca', w_pca)

lmdas = list(np.arange(0.2, 0.9, 0.01))

# penalty = select_penalty(y_obs, n=60, lmdas=lmdas, M = 30, sim_fn=GNK,
#                theta=true_params, shrinkage="warton", sum_fn=ss_robust,
#                 whitening=w_pca)
# print('penalty', penalty)
# print(1/0)


res = elfi.BSL(model['_summary'], batch_size=60, n_batches=1, y_obs=y_obs, n_sims=60, method="semiBsl",
               shrinkage="warton", penalty=0.3, n_obs=n_obs, #TODO: better penalty
               whitening=w_pca
               ).sample(2000,
               params0=np.array(true_params), sigma_proposals=est_post_cov)

toc = time.time()

np.save('A_data_110221gnk.npy', res.samples['A'])
np.save('B_data_110221gnk.npy', res.samples['B'])
np.save('g_data_110221gnk.npy', res.samples['g'])
np.save('k_data_110221gnk.npy', res.samples['k'])

print('totalruntime:', toc - tic)


print(res)
# print(np.cov)
res.plot_marginals(selector=None, bins=None, axes=None)
# res.plot_pairs()
plt.show()