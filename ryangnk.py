from elfi.examples.gnk import get_model, ss_robust, GNK
import elfi
import numpy as np
import matplotlib.pyplot as plt
import time
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
true_params = [3, 1, 2, 0.5] # a, b, g, k
np.random.seed(123)

n_obs=500

model = get_model(n_obs=n_obs, true_params=true_params)

# def sim_gnk(x):
    # GNK

# sim = GNK(n_obs=n_obs, *true_params)
# print('sim', sim.shape)
# sum_stat1 = np.array([sim[i, 0, :] for i in range(1000)])
# sum_stat1 = sum_stat1.flatten()
# plt.hist(sum_stat1, density=True, bins=30)
# plt.show()
# print('sum_stat1', sum_stat1)
# print(1/0)
# sim = sim.flatten()
# y_obs = ss_robust(sim)
# y_obs = y_obs.flatten()
# print('y_obs', y_obs)

# tic = time.time()
# lmdas = list(np.exp(np.arange(-5.5, 1.5, 0.1)))

est_post_cov = np.array(
    [[ 0.00058258,  0.00044668, -0.0002838,  -0.00022737],
 [ 0.00044668,  0.00256855,  0.00292011, -0.00284012],
 [-0.0002838,   0.00292011,  0.01953341, -0.00134999],
 [-0.00022737, -0.00284012, -0.00134999,  0.00590356]])

# sim_mat = np.zeros((1000, 4))  # TODO: Magic numbers...
# for i in range(1000):
#     sim_mat[i, :] = ss_robust(GNK(n_obs=n_obs, *true_params)).flatten() # obs as row


w_pca = estimate_whitening_matrix(model, theta_point=true_params,
                 summary_names=['ss_robust'], batch_size=10000)
print('w_pca', w_pca)

batch_size = 200

lmdas = list(np.arange(0.2, 0.9, 0.02))
penalty = select_penalty(n=batch_size, lmdas=lmdas, model=model, M = 20,
               theta=true_params, shrinkage="warton",
                whitening=w_pca, summary_names=['ss_robust'])
print('penalty', penalty)
# print(1/0)

res = elfi.BSL(model, batch_size=batch_size, #n_batches=1, n_sims=60, #method="semiBsl",
               shrinkage="warton", 
               penalty=penalty, 
               n_obs=n_obs, #TODO: better penalty
               chains=1, chain_length=10000,
               burn_in=1000,
               whitening=w_pca,
               summary_names=['ss_robust'],
            #    params0=np.array(true_params)
               ).sample(
                    10000,
                    sigma_proposals=0.1*np.eye(4),
                    params0=np.array([6, 5, 8, 5]))
            #    params0=np.array(true_params), sigma_proposals=10*est_post_cov)

# toc = time .time()

np.save('A_data_110221gnk.npy', res.samples['A'])
np.save('B_data_110221gnk.npy', res.samples['B'])
np.save('g_data_110221gnk.npy', res.samples['g'])
np.save('k_data_110221gnk.npy', res.samples['k'])

# print('totalruntime:', toc - tic)


print(res)
# print(np.cov)
res.plot_marginals(selector=None, bins=None, axes=None)

plt.savefig("gnk_marginals.png")

# res.plot_pairs()
# plt.show()