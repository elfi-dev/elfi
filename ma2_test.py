import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from elfi.examples import ma2
from elfi.methods.bsl.select_penalty import select_penalty
from elfi.methods.bsl.estimate_whitening_matrix import \
    estimate_whitening_matrix
from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood
import elfi
import time


# def t_student_sl(ssx, observed=None, *args, **kwargs):
#     sample_mean = ssx.mean(0)
#     n, ns = ssx.shape
#     sample_cov = np.atleast_2d(np.cov(ssx, rowvar=False))
#     loglik = ss.multivariate_t.logpdf(
#             observed,
#             loc=sample_mean,
#             shape=sample_cov,
#             df=n
#             )
#     return loglik  # + self.prior.logpdf(x)

def dummy_func(x):
    return x


def run_ma2():
    # np.random.seed(132)

    true_params = np.array([0.6, 0.2])

    m = ma2.get_model(n_obs=100, seed_obs=4)
    # eps = 2.0
    # delta = 1.0
    elfi.Summary(dummy_func, m['MA2'], name='identity')
    elfi.SyntheticLikelihood("ubsl",
                             m['identity'],
                            #  adjustment="mean",
                             name="SL")
    batch_size = 500
    # summary_names = ['identity', 'S1', 'S2']
    # m['SL'].become(elfi.SyntheticLikelihood("bsl", m['S1'], m['S2'], m['identity']))

    # elfi.SyntheticLikelihood("bsl", m['identity'], name="SL")

    # W = estimate_whitening_matrix(m['SL'], true_params,
    #                               batch_size=20000,
    #                               seed=1)
    # print('W', W)
    # m['SL'].become(elfi.SyntheticLikelihood("semibsl",
    #                                         m['identity'], whitening=W,
    #                                         shrinkage="warton"))

    # elfi.SyntheticLikelihood("semiBsl", m['identity'], whitening=W, shrinkage="warton",
    #                          name="semiSL")

    # batch_size = np.array([50, 100])

    # penalty = select_penalty(batch_size=batch_size,
    #                          M=10,
    #                         #  method="bsl",
    #                          shrinkage="glasso",
    #                         #  whitening=W,
    #                          sigma=1.2,
    #                          theta=[0.6, 0.2],
    #                          model=m,
    #                          discrepancy_name='SL',
    #                          verbose=True
    #                          )
    # penalty = 0.3
    # m['SL'].become(elfi.SyntheticLikelihood("bsl",  #  m['S1'], m['S2'],
    #         m['identity'],# whitening=W,
    #         shrinkage="glasso", penalty=penalty))

    # pool = elfi.ArrayPool(['t1', 't2', 'identity', 'SL'])
    bsl_obj = elfi.BSL(
                m['SL'],
                batch_size=batch_size,
                # pool=pool
                seed=123,
                )
    # M = 100
    # log_SL = bsl_obj.log_SL_stdev(true_params, batch_size, M)
    # print('log_SL', log_SL)
    # bsl_obj.plot_correlation_matrix(true_params, batch_size=10000, precision=True)
    # plt.savefig("plot_precision_ma2.png")

    # bsl_obj.plot_summary_statistics(batch_size=200000, theta_point=true_params)
    # plt.savefig("plot_summaries.png")
    # print(1/0)
    tic = time.time()
    n_samples = 2000
    bsl_res = bsl_obj.sample(
                    n_samples,
                    sigma_proposals=np.array([[0.02, 0.01],
                                              [0.01, 0.02]]),
                    params0=true_params,
                    burn_in=0
                )
    toc = time.time()
    print('time: ', toc - tic)
    ess = bsl_res.compute_ess()
    print(ess)
    print('bsl_res', bsl_res)
    bsl_res.plot_traces()
    reference_value = {'t1': 0.6, 't2': 0.2}
    plt.savefig("plot_traces_ubsl.png")
    bsl_res.plot_marginals(bins=30, reference_value=reference_value)
    plt.savefig("plot_marginals_ubsl.png")
    bsl_res.plot_pairs(bins=30, reference_value=reference_value)
    plt.savefig("plot_pairs_ubsl.png")


if __name__ == '__main__':
    run_ma2()
