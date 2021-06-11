import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import daycare
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix

import elfi

np.random.seed(123)
batch_size = 1

true_params = [3.6, 0.6, 0.1]

m = daycare.get_model(true_params=true_params)

summary_names = ['Shannon', 'n_strains', 'prevalence', 'multi']
observed_ss = np.array([m[summary_name].observed for summary_name in summary_names])
ssy = observed_ss.flatten()


W = estimate_whitening_matrix(model=m, summary_names=summary_names, batch_size=120)
print('W W W W ', W )
np.save("est_whitening_mat.npy", W)

lmdas = list(np.arange(0.3, 0.8, 0.02))
penalty = select_penalty(ssy=ssy, n=batch_size, lmdas=lmdas, M=10, sigma=1.2,
                         theta=[0.6, 0.2], shrinkage="warton",
                         summary_names=summary_names,
                         model=m,
                         whitening=W
                         )

print('penalty', penalty)

print('observed_ss', observed_ss)
print(1/0)

res = elfi.BSL(m, y_obs=ssy, summary_names=summary_names, method="bsl",
            #    simulation_names=['MA2'],
                penalty=0.46, shrinkage="warton",
               batch_size=batch_size, chains=1, chain_length=5000, burn_in=100,
               sigma_proposals=np.array([[0.2, 0.1], [0.1, 0.2]]),
               whitening=W)


