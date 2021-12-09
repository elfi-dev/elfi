
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as ss

import elfi
from elfi.examples import bdm

# Make an external command. {0} {1} are positional arguments and {seed} a keyword argument `seed`.
# command = 'echo {0} {1} {seed}'
# echo_sim = elfi.tools.external_operation(command)

# Test that `echo_sim` can now be called as a regular python function
# echo_sim(3, 1, seed=123)

# Fixed model parameters
# delta = 0
# tau = 0.198
# N = 20

# The zeros are to make the observed population vector have length N
y_obs = np.array([6, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int16')
# src_path = elfi.examples.bdm.get_sources_path()
# os.system('make -C $/Users/ryankelly/python_projects/elfi/elfi/examples/cpp')

# make -C elfi/examples/cpp  # TODO: use this to make bdm in main (delete tho)

m = bdm.get_model()
# print('dir', dir(m))
# elfi.Prior('uniform', .005, 2, model=m, name='alpha')
seed = 20170511
batch_size = 100
bsl_res = elfi.BSL(m,
    summary_names=['T1'],
    batch_size=batch_size,
).sample(1000, sigma_proposals=np.eye(1), params0=np.array([1]))

est_cov_mat = bsl_res.get_sample_covariance()
print('est_cov_mat', est_cov_mat)
bsl_res.plot_marginals(bins=50)
# np.save("bsl_res.npy", bsl_res ) # TODO: correctly
plt.savefig("bdm_marginals_identity.png")
bsl_res.plot_pairs(bins=50)
plt.savefig("bdm_pairs_identity.png")
