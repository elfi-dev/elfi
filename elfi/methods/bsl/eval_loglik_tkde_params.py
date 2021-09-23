import numpy as np
from elfi.methods.bsl.hyperbolic_power_transformation import \
    hyperbolic_power_transformation


def sech(x):
    """Helper function for transformation KDE

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 1/(np.cosh(x))


def eval_loglik_tkde_params(betas, s):
    # TODO: psi, lmda ? from where?
    psi, lmda = betas[0], betas[1]
    psi = np.exp(psi)

    lmda_tmp = 1
    if not np.isposinf(lmda):
        lmda_tmp = (np.exp(lmda) - 1) / \
                        (1 + np.exp(lmda))
    lmda = lmda_tmp

    x = s
    x = np.array(x)
    nu_hat = 1/np.sqrt(np.mean(hyperbolic_power_transformation(x, 1, psi, lmda) ** 2))
    # res = 0.5 * np.sum(nu_hat * hyperbolic_power_transformation(x, nu_hat, psi, lmda)) - \
    #     len(x) * np.log(nu_hat) - \
    #     np.sum(np.log(1 - lmda*(np.tanh(psi*x)**2))) - \
    #     (lmda-1)*np.sum(np.log(sech(psi*x)))  # TODO: Psi in Tsai meaning?
    res2 = (0.5*np.sum((nu_hat*np.sinh(psi*x)*
        (sech(psi*x) ** (lmda))/psi)**2)-
        len(x)*np.log(nu_hat)-
        np.sum(np.log(1-(lmda)*np.tanh(psi*x)**2))-
        ((lmda)-1)*np.sum(np.log(sech(psi*x))))
    # res = np.abs(res)  # TODO: safe?
    # print('res', res)
    # print('res2', res2)
    # print(1/0)
    return res2