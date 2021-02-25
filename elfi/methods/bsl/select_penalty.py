from elfi.methods import posteriors
import numpy as np
import elfi.methods.bsl.pdf_methods as pdf
from collections import namedtuple

def select_penalty(ssy, n, lmdas, M, sigma=1.5, theta = None,
                   method="bsl", shrinkage="glasso",
                   sim_fn=None, sum_fn=None, whitening=None, *args, **kwargs):
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
    ns = ssy.size
    n_lambda = len(lmdas)
    print('n_lambda', n_lambda)
    print('NNNN', n)
    print('MMMM', M)
    #nMax?  # TODO: extend to allow for a collection of n values

    print('selectingpenalty')

    # sim_fn = self.model.get_node('_simulator')['attr_dict']['_operation']
    # sum_fn = self.model.get_node('_summary')['attr_dict']['_operation']

    # add attributes used when calculating log-likelihood
    # self.y_obs = ssy
    # self.prior.logpdf = 0  # TODO: this doesn't need to be computed cause only at same point?

    logliks = np.zeros((M, n_lambda))

    def dummy_func(x):
        return 0

    DummyPrior = namedtuple('DummyPrior', 'logpdf')
    DummySelf = namedtuple('DummySelf', 'y_obs prior')
    dummy_prior = DummyPrior(logpdf=dummy_func)
    dummy = DummySelf(y_obs=ssy, prior=dummy_prior)

    print()
    for m in range(M):
        # print('MMiter', m)
        print('randomnumber', np.random.randint(0, 10))
        sims = sim_fn(*theta, batch_size=n, **kwargs) # TODO: distinguish kwargs?
        print('sims', sims.shape)
        ssx = sum_fn(sims).reshape((300, 100)) # TODO: ROBUST
        # for i in range(n):
        for k in range(n_lambda):
            lmdacurr = lmdas[k]

            # TODO: lowercase? - copied across from posteriors
            if method == "bsl":
                logliks[m, k] = \
                    pdf.gaussian_syn_likelihood(dummy, theta, ssx, shrinkage, lmdas[k], whitening)
            elif method == "semiBsl":
                logliks[m, k] = pdf.semi_param_kernel_estimate(theta, ssx)

    # choose the lambda with the empirical s.d. of the log SL estimates
    # closest to sigma
    print('logliks', logliks)
    std_devs = np.array([np.std(logliks[:, i]) for i in range(n_lambda)]) # TODO: CHECK AXIS
    print('std_devs', std_devs)
    closest_lmda = np.min(np.abs(std_devs - sigma))
    closest_arg = np.argmin(np.abs(std_devs - sigma))
    print('closest_arg', closest_arg, 'lamm?', lmdas[closest_arg])
    print('closest_lmda', closest_lmda)
    # closest_lmda = np
    return lmdas[closest_arg]