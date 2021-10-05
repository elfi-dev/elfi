import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples.astro.astro import get_model
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
import time

import elfi


def run_astro():
    np.random.seed(1)
    m = get_model(seed_obs=123)
    n_samples = 10000
    summary_names = ['identity']
    batch_size = 100  #! TODO
    seed=4
    true_params = np.array([0.7, 0.3, -1.0])
    logitTransformBound = np.array([
                                    [-1, 3],
                                    [-1e+2, 1e+2],
                                    [0, 1]])
    #  
    # penalty = select_penalty(batch_size=batch_size, lmdas=np.array([0]),
    #                      M=5, sigma=1.2,
    #                      theta=true_params, shrinkage="warton",
    #                      summary_names=summary_names,
    #                      seed=seed,
    #                     #  whitening=W,
    #                      method="semibsl",
    #                      model=m,
    #                      verbose=True
    #                      )

    # print(penalty)
    
    bsl_obj = elfi.BSL(
        m,
        summary_names=summary_names,
        method="semibsl",
        batch_size=batch_size,
        seed=4,
        # parameter_names=['om', 'w0', 'h0']
        # logitTransformBound=logitTransformBound
    )

    # bsl_obj.plot_summary_statistics(batch_size=4000, theta_point=true_params)
    # plt.savefig("plot_summaries_astro.png")
    tic = time.time()
    bsl_res = bsl_obj.sample(n_samples,
                             sigma_proposals=0.001*np.eye(3),
                            #  params0=true_params
                             )
    toc = time.time()
    print('time', toc-tic)
    print('bsl_res', bsl_res)
    bsl_res.plot_traces()
    plt.savefig("plot_traces_astro.png")
    bsl_res.plot_marginals(bins=30)
    plt.savefig("plot_marginals_astro.png")
    bsl_res.plot_pairs(bins=30)
    plt.savefig("plot_pairs_astro.png")


if __name__ == '__main__':
    run_astro()