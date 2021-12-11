import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import misspecified_ma1
import time

import elfi


def run_misspec_ma1():
    np.random.seed(123)

    # true_params = [1]
    summary_names = ['S1', 'S2']
    # true_params = [-0.736, 0.9, 0.36]

    batch_size = 50
    mcmc_iters = 100000
    burn_in = 10000
    num_runs = 50
    num_params = 1

    res_mat_sbsl = np.zeros((mcmc_iters - burn_in, num_params, num_runs))
    res_mat_rbslm = np.zeros((mcmc_iters - burn_in, num_params, num_runs))
    res_mat_rbslv = np.zeros((mcmc_iters - burn_in, num_params, num_runs))

    for i in range(num_runs):
        # tic = time.time()
        m = misspecified_ma1.get_model(n_obs=100, seed_obs=i)
        # d = elfi.SyntheticLikelihood("bsl", m['S1'], m['S2'], name="SL")
        # bsl_res = elfi.BSL(
        #             m['SL'],
        #             batch_size=batch_size,
        #             ).sample(
        #                 mcmc_iters,
        #                 burn_in=burn_in,
        #                 sigma_proposals=0.1*np.eye(1),
        #                 params0=[0]
        #             )

        # print('bsl_res', bsl_res)
        elfi.SyntheticLikelihood("misspecbsl", m['S1'], m['S2'],
                                 type_misspec="mean", name="R_BSL_M")
        # # bsl_res.plot_marginals(bins=30)
        # res_mat_sbsl[:, 0, i] = bsl_res.outputs['t1']

        bsl_res_misspec_mean = elfi.BSL(
                    m['R_BSL_M'],
                    batch_size=batch_size,
                    ).sample(
                        mcmc_iters,
                        burn_in=burn_in,
                        sigma_proposals=0.1*np.eye(1),
                        params0=[0]
                    )
        print('bsl_res_misspec_mean', bsl_res_misspec_mean)
        res_mat_rbslm[:, 0, i] = bsl_res_misspec_mean.outputs['t1']
        elfi.SyntheticLikelihood("misspecbsl", m['S1'], m['S2'],
                                 type_misspec="variance", name="R_BSL_V")

        bsl_res_misspec_var = elfi.BSL(
                     m['R_BSL_V'],
                    batch_size=batch_size,
                    ).sample(
                        mcmc_iters,
                        burn_in=burn_in,
                        sigma_proposals=0.1*np.eye(1),
                        params0=[0]
                    )
        print(bsl_res_misspec_var)
        res_mat_rbslv[:, 0, i] = bsl_res_misspec_var.outputs['t1']
        ess = bsl_res_misspec_var.compute_ess()
        print('ess: ', ess)
        # bsl_res_misspec_var.plot_traces()
        # plt.savefig("plot_traces_misspec.png")
        # bsl_res_misspec_var.plot_marginals()
        # plt.savefig("plot_marginals_misspec.png")
        # bsl_res_misspec_var.plot_pairs()
        # plt.savefig("plot_pairs_misspec.png")
        # print('bsl_res_misspec_var', bsl_res_misspec_var)
        # res_mat_rbslv[:, 0, i] = bsl_res_misspec_var.outputs['t1']

    # plt.savefig("plot_marginals_standard_contaminated.png")

    # plot standard BSL
    t1_samples = res_mat_sbsl.flatten()
    plt.hist(t1_samples, bins=50)
    plt.savefig("plot_marginals_contaminated_ma1_sbsl.png")
    plt.clf()
    # plot R-BSL-M
    t1_samples = res_mat_rbslm.flatten()
    plt.hist(t1_samples, bins=50)
    plt.savefig("plot_marginals_contaminated_ma1_rbslm.png")
    plt.clf()

    # plot R-BSL-V
    t1_samples = res_mat_rbslv.flatten()
    plt.hist(t1_samples, bins=50)
    plt.savefig("plot_marginals_contaminated_ma1_rbslv.png")
    # plt.show()    


if __name__ == '__main__':
    run_misspec_ma1()

