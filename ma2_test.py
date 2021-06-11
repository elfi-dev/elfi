import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import ma2
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix

import elfi

np.random.seed(123)

true_params = np.array([0.6, 0.2])

# def dummy_func(x, batch_size=batch_size):
#     return x.reshape(batch_size, -1)

# W = estimate_whitening_matrix(m, summary_names, true_params, batch_size=20000) #, method="semiBsl") # summary same as simulation here
m = ma2.get_model(n_obs=50, seed_obs=1)
n_samples = 50000
summary_names = ['identity']
# W = estimate_whitening_matrix(m, summary_names, true_params, batch_size=10000)
batch_size = 500
# lmdas = list(np.exp(np.arange(-5.5, -1.5, 0.2)))
# penalty = select_penalty(batch_size=batch_size, lmdas=lmdas, M=30, sigma=1.2,
#                          theta=[0.6, 0.2], shrinkage="glasso",
#                          summary_names=["identity"],
#                          method="bsl",
#                          model=m
#                          )

bsl_res = elfi.BSL(
            m,
            summary_names=summary_names,
            method="bslmisspec",
            batch_size=batch_size,
            type_misspec="variance",
            burn_in=10,
            # penalty=penalty,
            # shrinkage="glasso"
            ).sample(
                  n_samples,
                  sigma_proposals=np.array([[0.02, 0.01],
                                            [0.01, 0.02]]),
                  params0=true_params
            )
print('bsl_res', bsl_res)
bsl_res.plot_marginals(bins=30)

# np.save('ma2_res.npy', bsl_res)

# res_bsl = res.s ample(500, params0=true_params)

# print('res_bsl', bsl_inf.extract_result())

print('bsl_res', bsl_res)

bsl_res.plot_marginals(bins=30)

plt.savefig("plot_marginals_misspec.png")

bsl_res.plot_pairs(bins=30)

plt.savefig("plot_pairs_misspec.png")
