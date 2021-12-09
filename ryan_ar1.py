import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import ar1
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix

import elfi


def run_ar1():
    m = ar1.get_model()
    summary_names = ['identity']
    true_params = [.9]

    # fake_true_params = [.75]

    # W = estimate_whitening_matrix(m, summary_names, true_params, batch_size=200000)
    # lmdas = list((np.arange(0.2, 0.9, 0.02)))

    batch_size = 6000

    # penalty = select_penalty(m['SL'], batch_size=batch_size, M=10, sigma=1.5,
                            #  theta=true_params, #shrinkage="glasso",
                            #  summary_names=["identity"],
                            # #  whitening=W,
                            # #  method="semibsl",
                            #  model=m
                            #  )
    # print('penalty_start', penalty)

    tic = time.time()
    bsl_res = elfi.BSL(
        m['SL'],
        summary_names=summary_names,
        batch_size=batch_size,
    ).sample(
        5000,
        sigma_proposals=0.001178,
        burn_in=250,
        params0=true_params
    )
    toc = time.time()
    print("time: ", toc - tic)
    print(bsl_res)

    est_cov_mat = bsl_res.get_sample_covariance()
    print('est_cov_mat', est_cov_mat)

    bsl_res.plot_marginals(bins=50)
    plt.savefig("ar1.png")


if __name__ == '__main__':
    run_ar1()
