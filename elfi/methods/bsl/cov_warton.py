import numpy as np
from .gaussian_copula_density import p2P
def cov_warton(S, gamma):
    print('gamma', gamma)
    if gamma < 0 or gamma > 1:
        raise("Gamma must be between 0 and 1")
    # print('S', S.shape)
    _, ns = S.shape
    eps = 1e-5 # prevent divide by 0 for r_hat
    print('SSS', S)
    D1 = np.diag(1/np.sqrt(np.diag(S + eps)))
    D2 =  np.diag(np.sqrt(np.diag(S + eps)))
    r_hat = np.matmul(np.matmul(D1, S), D1)
    # print('(1 - gamma)*np.eye(ns)', (1 - gamma)*np.eye(ns))
    print('r_hat', r_hat)
    r_hat_gamma = gamma * r_hat + (1 - gamma)*np.eye(ns)
    print('r_hat_gamma', r_hat_gamma)
    Sigma = np.matmul(np.matmul(D2, r_hat_gamma), D2)
    # print('Sigma', Sigma.shape)
    # print(1/0)
    return Sigma

def corr_warton(R, gamma, dim):
    """[summary]

    Args:
        R ([type]): [description]
        gamma ([type]): [description]
        dim ([type]): [description]

    Returns:
        [type]: [description]
    """
    _, ns = R.shape
    return gamma * R + (1 - gamma) * np.eye(ns)

