from elfi.examples.gnk import get_model, ss_robust, GNK
import elfi
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats as ss

true_params = [3, 1, 2, 0.5] # a, b, g, k
np.random.seed(123)

model = get_model(n_obs=50, true_params=true_params)

# def sim_gnk(x):
    # GNK
num_sims = 2000
sum_mat = np.zeros((num_sims, 4))

for i in range(num_sims):
    sim = GNK(n_obs=10000, *true_params)
    sum_stat = ss_robust(sim).flatten()
    sum_mat[i, :] = sum_stat

for j in range(4):
    xs = np.linspace(min(sum_mat[:, j]), max(sum_mat[:, j]))
    kde = ss.gaussian_kde(sum_mat[:, j])
    plt.plot(xs, kde(xs))
    plt.show()

# sim = GNK(*true_params)
# sum_stat = ss_octile(sim)
# print('sum_stat', sum_stat)

