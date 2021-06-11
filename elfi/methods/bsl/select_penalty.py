from elfi.methods import posteriors
import numpy as np
import elfi.methods.bsl.pdf_methods as pdf
from collections import namedtuple

import elfi

def select_penalty(batch_size, lmdas, M, ssy=None, model=None, sigma=1.5, theta=None,
                   method="bsl", shrinkage="glasso", summary_names=None,
                   whitening=None, *args, **kwargs):
    """[summary]

    Args:
        ssy ([type]): [description]
        n ([type]): [description]
        lmda ([type]): [description]
        M ([type]): [description]
        model ([type]): [description]
        theta ([type], optional): [description]. Defaults to None.
        method (list, optional): [description]. Defaults to ["BSL", "semiBsl"].
        shrinkage (list, optional): [description]. Defaults to ["glasso"].

    Returns:
        [type]: [description]
    """
    # if ssy is None:
    #     ssy = model[]

    ssy = np.array(list(model.observed.values())).flatten()

    shrinkage = None



    ns = ssy.size
    n_lambda = len(lmdas)
    if ssy.ndim > 1:
        ssy = model.observed.flatten()
    #nMax?  # TODO: extend to allow for a collection of n values

    print('selectingpenalty')

    # sim_fn = self.model.get_node('_simulator')['attr_dict']['_operation']
    # sum_fn = self.model.get_node('_summary')['attr_dict']['_operation']

    # add attributes used when calculating log-likelihood
    # self.y_obs = ssy
    # self.prior.logpdf = 0  # TODO: this doesn't need to be computed cause only at same point?

    logliks = np.zeros((M, n_lambda))

    # def dummy_func(x):
    #     return 0

    # DummyPrior = namedtuple('DummyPrior', 'logpdf') # TODO? weird?
    # DummySelf = namedtuple('DummySelf', 'y_obs prior')
    # dummy_prior = DummyPrior(logpdf=dummy_func)
    # dummy = DummySelf(y_obs=ssy, prior=dummy_prior)

    # model.generate()

    for m_iteration in range(M):
        # prior = model.generate(1, model.parameter_names)
        # sum_batch = model.generate(n, model)
        # print('sum_batch', sum_batch)
        # output_names =  model.parameter_names + summary_names
        # batch = model.generate(n, output_names)
        # print('batch', batch)
        # # sum_batch = model.update()
        # for summary in sum_batch.items():
        #     print('summary', summary[1])
        # print('sum_batch', sum_batch)
        # ssx = np.array(list(zip(sum_batch.items())))
        # print('ssx', ssx.shape)
        # s1, s2 = ssx.shape[0:2]
        # ssx = ssx.reshape((s1, s2))
        # print('ssx', ssx)
        # sims = sim_fn(*theta, batch_size=n, **kwargs) # TODO: distinguish kwargs?
        # ssx = sum_fn(sims).reshape((n, len(ssy)))
        # for i in range(n):
        for k in range(n_lambda):
            m = model.copy()
            bsl_temp = elfi.BSL(m, summary_names=summary_names, method=method,
                    batch_size=batch_size, chains=1, chain_length=1, burn_in=0,
                    penalty=lmdas[k], shrinkage=shrinkage, whitening=whitening
                    )
            loglik = bsl_temp.select_penalty_helper(theta)
            print('loglik', loglik)
            print('m', m_iteration, 'k', k)
            logliks[m_iteration, k] = loglik
            # lmdacurr = lmdas[k]
            # # TODO: lowercase? - copied across from posteriors
            # if method == "bsl":
            #     logliks[m, k] = \
            #         pdf.gaussian_syn_likelihood(model, theta, ssx, shrinkage, lmdas[k], whitening)
            # elif method == "semiBsl":
            #     logliks[m, k] = pdf.semi_param_kernel_estimate(model, theta, ssx, shrinkage, lmdas[k], whitening)

    # choose the lambda with the empirical s.d. of the log SL estimates
    # closest to sigma
    print('logliks', logliks)
    std_devs = np.array([np.std(logliks[:, i]) for i in range(n_lambda)]) # TODO: CHECK AXIS
    print('std_devs', std_devs)
    print('lmdas', lmdas)
    closest_lmda = np.min(np.abs(std_devs - sigma))
    closest_arg = np.argmin(np.abs(std_devs - sigma))
    print('closest_arg', closest_arg, 'lamm?', lmdas[closest_arg])
    # print('closest_lmda', closest_lmda)
    # closest_lmda = np
    return lmdas[closest_arg]