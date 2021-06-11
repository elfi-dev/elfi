import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import ricker
from elfi.methods.bsl.select_penalty import select_penalty

import elfi

np.random.seed(123)

true_param = [3.8, 10, 0.3]
batch_size = 1000

m = ricker.get_model()

summary_names = ['Mean', 'Var', '#0']

batch_size=100

lmdas = list(np.exp(np.arange(-5.5, -1.5, 0.2)))

# y_obs = m['observed']

# observed_ss = np.array([m[summary_name].observed for summary_name in summary_names])
# print('observed_ss', observed_ss)

penalty = select_penalty(batch_size=batch_size, lmdas=lmdas, M=10, sigma=1.2,
                         theta=true_param, shrinkage="glasso",
                         summary_names=summary_names,
                         method="bsl",
                         model=m
                         )

# penalty = select_penalty(ssy=observed_ss, n=batch_size, lmdas=lmdas, M=10, sigma=1.2,
#                          theta=[0.6, 0.2], shrinkage="glasso",
#                          summary_names=summary_names,
#                          model=m
#                         #  sim_fn=MA2, sum_fn=dummy_func, 
#                         #  whitening=W
#                          )
print('penalty', penalty)
# print('W' ,W )
print(1/0)

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