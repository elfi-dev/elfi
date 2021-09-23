"""Helper to pick the penalty for glasso and warton shrinkage"""

from elfi.methods import posteriors
import numpy as np
import elfi.methods.bsl.pdf_methods as pdf
from collections import namedtuple
import warnings

import elfi


def select_penalty(batch_size, lmdas, M, ssy=None, model=None, sigma=1.5, theta=None,
                   method="bsl", shrinkage="glasso", summary_names=None,
                   whitening=None, seed=None, *args, **kwargs):
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

    n_lambda = len(lmdas)
    batch_size = np.array([batch_size]).flatten()
    ns = len(batch_size)

    # TODO: different default lmdas depending on penalty method

    logliks = np.zeros((M, ns, n_lambda))

    for m_iteration in range(M):
        for i in range(ns):
            for k in range(n_lambda):
                # np.random.RandomState(m_iteration*1000 + k)  # arbitrary seed
                seed = m_iteration*1000 + k
                m = model.copy()
                # try:
                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore")
                bsl_temp = elfi.BSL(m, summary_names=summary_names, method=method,
                        batch_size=batch_size[i],
                        penalty=lmdas[k], shrinkage=shrinkage, whitening=whitening,
                        seed=seed
                        )
                loglik = bsl_temp.select_penalty_helper(theta)
                # except FloatingPointError:  # poor penalty + number combination
                    # loglik = np.NINF
                logliks[m_iteration, i, k] = loglik

    # choose the lambda with the empirical s.d. of the log SL estimates
    # closest to sigma
    closest_lmdas = np.zeros(ns)
    for i in range(ns):
        std_devs = np.array([np.std(logliks[:, i, j]) for j in range(n_lambda)]) # TODO: CHECK AXIS
        closest_lmda = np.min(np.abs(std_devs - sigma))
        closest_arg = np.argmin(np.abs(std_devs - sigma))
        closest_lmdas[i] = lmdas[closest_arg]

    return closest_lmdas