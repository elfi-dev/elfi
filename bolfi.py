import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

import logging
# logging.basicConfig(level=logging.INFO)

# Set an arbitrary global seed to keep the randomly generated quantities the same
seed = 1
np.random.seed(seed)

import elfi


from elfi.examples import ma2
model = ma2.get_model(seed_obs=seed)
elfi.draw(model)


log_d = elfi.Operation(np.log, model['d'])

bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=10,
                   bounds={'t1':(-2, 2), 't2':(-1, 1)}, acq_noise_var=[0.1, 0.1], seed=seed)

post = bolfi.fit(n_evidence=200)

bolfi.target_model

bolfi.plot_state();
bolfi.plot_discrepancy();
# post2 = bolfi.extract_posterior(-1.)
post.plot(logpdf=True)
result_BOLFI = bolfi.sample(1000)
result_BOLFI.plot_traces()
result_BOLFI.plot_marginals()
print(result_BOLFI)
plt.show()

