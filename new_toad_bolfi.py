import numpy as np

import elfi
import matplotlib.pyplot as plt

from elfi.examples import toad
from elfi.methods.bo.acquisition import ExpIntVar


def run_toad_bolfi():

    true_params = [1.7, 35.0, 0.6]

    m = toad.get_model(true_params=true_params, seed_obs=1, parallelise=False, n_cpus=1)

    pool = elfi.ArrayPool(['S', 'd'])
    # elfi.SyntheticLikelihood("semiBsl", m['S'], name="SL")
    bolfi = elfi.BOLFI(m['d'], batch_size=1, initial_evidence=20, update_interval=10,
                    bounds={'alpha': (1, 2), 'gamma': (0, 100), 'p0': (0, 0.9)},
                    acq_noise_var=[0.1, 0.1, 0.1], pool=pool)

    post = bolfi.fit(n_evidence=50)

    result_BOLFI = bolfi.sample(1000, warmup=0)
    print('result_BOLFI', result_BOLFI)

    result_BOLFI.plot_marginals()
    plt.savefig("toad_bolfi_marginals.png")

    result_BOLFI.plot_discrepancy()
    plt.savefig("toad_bolfi_discrepancy.png")

    result_BOLFI.plot_state()
    plt.savefig("toad_bolfi_state.png")



if __name__ == '__main__':
    run_toad_bolfi()