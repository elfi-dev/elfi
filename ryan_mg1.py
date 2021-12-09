import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import mg1
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix

import elfi


def run_mg1():
    m = mg1.get_model(n_obs=50, seed_obs=123)

    batch_size = 900
    true_params = [1., 5., 0.2]

    approx_est_cov = np.array([[ 0.01154655, -0.00107713, -0.00173222],
                            [-0.00107713,  0.03442745,  0.00058584],
                            [-0.00173222,  0.00058584,  0.01856123]])

    # lmdas = list((np.exp(np.arange(-5.5, -1.5, 0.2))))
    # lmdas = list((np.arange(0.3, 0.8, 0.02)))

    # penalty = select_penalty(batch_size=batch_size, lmdas=lmdas, M=30, sigma=1.2,
    #                          theta=true_params, shrinkage="glasso",
    #                          summary_names=summary_names,
    #                         #  whitening=W,
    #                          method="semibsl",
    #                          model=m
    #                          )

    # print('penalty', penalty)
    # print(1/0)
    # elfi.set_client('multiprocessing')
    mcmc_iterations =10000
    bsl_obj = elfi.BSL(
        m['SL'],
        batch_size=batch_size,
        seed=123
    )
    M = 100
    log_SL = bsl_obj.log_SL_stdev(true_params, batch_size, M)
    print('log_SL', log_SL)
    # bsl_obj.plot_summary_statistics(batch_size=20000, theta_point=true_params)
    # plt.savefig("plot_mg1_summaries.png")

    bsl_res = bsl_obj.sample(
        mcmc_iterations,
        sigma_proposals=approx_est_cov,
        # params0=true_params,
        burn_in=1000
    )

    print(bsl_res)
    # est_cov_mat = bsl_res.get_sample_covariance()
    # print('est_cov_mat', est_cov_mat)
    bsl_res.plot_traces()
    plt.savefig("mg1_traces.png")
    reference_value = {'t1': 1.0, 't2': 5.0, 't3': 0.2}
    bsl_res.plot_marginals(bins=30, reference_value=reference_value)
    # np.save("bsl_res.npy", bsl_res ) # TODO: correctly
    plt.savefig("mg1_marginals_logidentity_semibsl.png")
    bsl_res.plot_pairs(bins=30, reference_value=reference_value)
    plt.savefig("mg1_pairs_logidentity_semibsl.png")


# bsl_res.plot_pairs()
if __name__ == '__main__':
    run_mg1()

