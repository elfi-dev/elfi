import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import ricker
from elfi.methods.bsl.select_penalty import select_penalty

import elfi

np.random.seed(123)

true_params = [3.8, 0.3, 10.]

m = ricker.get_model(n_obs=100)

summary_names = ['Mean', 'Var', '#0']

batch_size = 10

param_cov_mat = 0.01*np.array([[2.25, -5.25, -1.35],
                               [-5.25, 25, 3],
                               [-1.35, 3, 2.25]])

# Test ABC-Rej as well
# rej = elfi.Rejection(m['d'], batch_size=100, seed=1).sample(1000)
# print('rej', rej)
# lmdas = list(np.exp(np.arange(-5.5, -1.5, 0.1)))

# penalty = select_penalty(batch_size=batch_size, lmdas=np.array([0]), M=30, sigma=1.2,
#                          theta=true_params, shrinkage="glasso",
#                          summary_names=summary_names,
#                          method="ubsl",
#                          model=m
#                          )

# print('penalty_start', penalty)
# print(1/0)

bsl_ricker = elfi.BSL(
            m,
            summary_names=summary_names,
            method="ubsl",
            batch_size=batch_size,
            # type_misspec="variance",
            burn_in=20000,
            # penalty=0.015,
            # shrinkage="glasso"
            )

# bsl_ricker.plot_summary_statistics(batch_size=2000, theta_point=true_params)
# plt.savefig("ricker_summaries.png")
# print(1/0)
bsl_res = bsl_ricker.sample(
                  200000,
                  sigma_proposals=param_cov_mat,
                  params0=true_params
            )

print(bsl_res)
bsl_res.plot_marginals(bins=50)
plt.savefig("ricker_marginals_uBsl.png")

bsl_res.plot_pairs(bins=50)
plt.savefig("ricker_pairs_uBsl.png")

bsl_res.plot_traces()
plt.savefig("ricker_trace_uBsl.png")

# y_obs = m['observed']

# observed_ss = np.array([m[summary_name].observed for summary_name in summary_names])
# print('observed_ss', observed_ss)


# penalty = select_penalty(ssy=observed_ss, n=batch_size, lmdas=lmdas, M=10, sigma=1.2,
#                          theta=[0.6, 0.2], shrinkage="glasso",
#                          summary_names=summary_names,
#                          model=m
#                         #  sim_fn=MA2, sum_fn=dummy_func, 
#                         #  whitening=W
#                          )
# print('penalty', penalty)
# print('W' ,W )
# print(1/0)

# sum_stats = m.generate(batch_size, outputs=summary_nodes)
# for sum_stat_key in sum_stats:
#     print('sum_stat_key', sum_stat_key)
#     sum_stat = sum_stats[sum_stat_key]
#     print('sum_stat', sum_stat)

#     minimum = min(sum_stat)
#     maximum = max(sum_stat)
#     kde = ss.gaussian_kde(sum_stat)
#     xs = np.linspace(minimum, maximum, 200)
#     plt.plot(xs, kde(np.log(xs)))
#     plt.show()