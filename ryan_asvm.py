import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import stochastic_volatility_model
from elfi.methods.bsl.select_penalty import select_penalty
import elfi

m = stochastic_volatility_model.get_model(seed_obs=123)
summary_names = ['identity']
batch_size = 20000
true_params = [1.2, 0.5]
asvm_bsl = elfi.BSL(
    m,
    summary_names=summary_names,
    batch_size=batch_size,
    shrinkage="glasso",
    penalty=0.1,
    standardise=True,
    # method="semiBsl",
    seed=3
)

# asvm_bsl.plot_summary_statistics(batch_size=20000, theta_point=true_params)
# plt.savefig("asvm_summaries.png")
# print(1/0)

bsl_res = asvm_bsl.sample(
    5,
    sigma_proposals=0.01*np.eye(2),
    params0=true_params
)

#TODO! curr - approx. 240 hours run sequential

print(bsl_res)
est_cov_mat = bsl_res.get_sample_covariance()
print('est_cov_mat', est_cov_mat)
bsl_res.plot_marginals(bins=50)
# np.save("bsl_res.npy", bsl_res ) # TODO: correctly
plt.savefig("asvm_marginals_identity.png")
bsl_res.plot_pairs(bins=50)
plt.savefig("asvm_pairs_identity.png")
