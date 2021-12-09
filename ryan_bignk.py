import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import bignk
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix

import elfi


def run_bignk():
    true_params = np.array([3, 4, 1, 0.5, 1, 2, .5, .4, 0.6])
    m = bignk.get_model()
    summary_names = ['ss_robust']

    batch_size = 200
    mcmc_iter = 300
    lmdas = list((np.arange(0.3, 0.8, 0.02)))
    # penalty = select_penalty(batch_size=batch_size, lmdas=np.array([1]), M=30, sigma=1.2,
    #                          theta=true_params, shrinkage="glasso",
    #                          summary_names=summary_names,
    #                         #  whitening=W,
    #                          method="semibsl",
    #                          model=m
    #                          )


    est_cov_mat = np.array([[ 0.00678746, 0.00031372, 0.00480741, 0.00101018, 0.00248509, -0.00691673, 
                            -0.00625261, -0.00353732, -0.00324986],
    [ 0.00031372, 0.00221448, 0.00172257, 0.00299738, 0.00075026, -0.01237923, 
        -0.00539615, -0.00523185, -0.00434071],
    [ 0.00480741, 0.00172257, 0.03041192, 0.00328968, 0.01559586, -0.02906985
    , -0.04141007, -0.01455316, 0.00919575],
    [ 0.00101018, 0.00299738, 0.00328968, 0.01440617, -0.00770499, 0.01175299
    , -0.01178046, -0.02767779, 0.00208513],
    [ 0.00248509, 0.00075026, 0.01559586, -0.00770499, 0.13345656, -0.05019588
    , -0.01091954, 0.01407991, 0.03645105],
    [-0.00691673, -0.01237923, -0.02906985, 0.01175299, -0.05019588, 0.2810845
    ,  0.05794362, -0.00117837, 0.05237798],
    [-0.00625261, -0.00539615, -0.04141007, -0.01178046, -0.01091954, 0.05794362
    ,  0.1081435,  0.042158, -0.01475752],
    [-0.00353732, -0.00523185, -0.01455316, -0.02767779, 0.01407991, -0.00117837
    ,  0.042158, 0.08098047, -0.00935728],
    [-0.00324986, -0.00434071, 0.00919575, 0.00208513, 0.03645105, 0.05237798
    , -0.01475752, -0.00935728, 0.24130605]])

    elfi.SyntheticLikelihood('bsl', m['ss_robust'], name='SL')

    bsl_bignk = elfi.BSL(
        m['SL'],
        summary_names=summary_names,
        batch_size=batch_size,
        # method="semiBsl",
        output_names=['BiGNK', 'ss_robust']
    )
    bsl_bignk.plot_correlation_matrix(true_params, batch_size=10000)
    # bsl_bignk.plot_summary_statistics(batch_size=2000, theta_point=true_params)
    plt.savefig("bignk_samplecov.png")
    bsl_res = bsl_bignk.sample(mcmc_iter, sigma_proposals=est_cov_mat, params0=true_params)
    est_cov_mat = bsl_res.get_sample_covariance()
    print('est_cov_mat', est_cov_mat)

    bsl_res.plot_marginals(bins=50)
    plt.savefig("bignk_marginals_semibsl.png")

    bsl_res.plot_pairs(bins=50)
    plt.savefig("bignk_pairs_identity_semibsl.png")

    bsl_res.plot_traces()
    plt.savefig("bignk_traces_semibsl.png")



if __name__ == '__main__':
    run_bignk()

