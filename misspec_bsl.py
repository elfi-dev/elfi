import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import contaminated_normal

import elfi

np.random.seed(123)

m = contaminated_normal.get_model(seed_obs=1)
true_params = [1]
summary_names = ['Mean', 'Var']

batch_size = 10
n_samples = 10000

bsl_res = elfi.BSL(
            m,
            summary_names=summary_names,
            observed=np.array([1, 1]),
            # method="bslmisspec",
            batch_size=batch_size,
            # type_misspec="mean",
            burn_in=10,
            
            # penalty=penalty,
            # shrinkage="glasso"
            ).sample(
                  n_samples,
                  sigma_proposals=[0.3],
                  params0=true_params
            )

print('bsl_res', bsl_res)

bsl_res.plot_marginals(bins=30)

plt.savefig("plot_marginals_standard_contaminated.png")
print(1/0)
bsl_res_misspec = elfi.BSL(
            m,
            summary_names=summary_names,
            method="bslmisspec",
            batch_size=batch_size,
            type_misspec="variance",
            burn_in=10,
            observed=np.array([1, 1]),
            # penalty=penalty,
            # shrinkage="glasso"
            ).sample(
                  n_samples,
                  sigma_proposals=np.eye(1),
                  params0=[0]
            )

print('bsl_res_misspec', bsl_res_misspec)

bsl_res_misspec.plot_marginals(bins=30)

plt.savefig("plot_marginals_misspec_contaminated.png")
