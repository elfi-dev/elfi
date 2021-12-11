import numpy as np
import elfi
import matplotlib.pyplot as plt
from elfi.examples import toad
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
from elfi.methods.bsl.select_penalty import select_penalty

import time


def run_toad():
    seed = 1234
    np.random.seed(seed)
    true_params = [1.7, 35.0, 0.6]

    m = toad.get_model(true_params=true_params, seed_obs=123)
    # true_params = [1.2, 27.0, 0.3]

    summary_names = ['S']
    batch_size = 504
    mcmc_iterations = 20000
    # elfi.set_client('multiprocessing')

    # W = estimate_whitening_matrix(m, summary_names, true_params, batch_size=20000)
    # np.save('toad_whitening.npy', W)
    # lmdas = list((np.arange(0.2, 0.9, 0.02)))
    W = np.load("est_whitening_mat.npy")
    penalty = select_penalty(model=m['SL'],
                             batch_size=np.array([50, 100, 500]), #lmdas=np.array([0]),
                             M=10, sigma=1.2,
                             theta=true_params, shrinkage="warton",
                             summary_names=summary_names,
                             seed=seed,
                             whitening=W,
                             method="bsl"
                             )
    # print('penalty', penalty)
    # toad_pool = elfi.OutputPool(['alpha', 'gamma', 'p0', 'S'])
    logitTransformBound = np.array([[1, 2],
                                    [0, 100],
                                    [0, 0.9]
                                    ])
    
    # elfi.SyntheticLikelihood("semiBsl", m['S'], name="SL")
    
    toad_bsl = elfi.BSL(m["SL"],
                        # n_batches=4,
                        batch_size=batch_size,
                        # whitening=W,
                        # penalty=penalty,
                        seed=seed,
                        # method="semibsl",
                        # pool=toad_pool
                        # type_misspec="variance"
                        )
    # toad_bsl.plot_summary_statistics(batch_size=5000, theta_point=true_params)
    # plt.savefig("toad_summaries.png")


    est_posterior_cov = np.array([[0.081, 0.007, 0.001],
    [0.007, 0.003, 0.001],
    [0.001, 0.001, 0.003]])

    tic = time.time()

    bsl_res = toad_bsl.sample(mcmc_iterations,
                            params0=true_params,
                            sigma_proposals=est_posterior_cov,
                            logitTransformBound=logitTransformBound,
                            burn_in=1000)
    toc = time.time()
    np.save("logposterior_toad.npy", toad_bsl.state['logposterior'])
    # toad_pool.save()

    print('time: ', toc - tic)
    print('bsl_res', bsl_res)

    bsl_res.plot_traces()
    plt.savefig("toad_traces_semiBsl.png")

    bsl_res.plot_marginals(bins=30)
    plt.savefig("toad_marginals_semiBsl.png")
    
    bsl_res.plot_pairs(bins=30)
    plt.savefig("toad_pairs_semiBsl.png")

    est_cov_mat = bsl_res.get_sample_covariance()
    print('est_cov_mat', est_cov_mat)


if __name__ == '__main__':
    run_toad()