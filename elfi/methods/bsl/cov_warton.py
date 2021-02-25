import numpy as np

def cov_warton(S, gamma):
    if gamma < 0 or gamma > 1:
        raise("Gamma must be between 0 and 1")
    # print('S', S.shape)
    _, ns = S.shape
    eps = 1e-5 # prevent divide by 0 for r_hat
    r_hat = np.diag(1/np.sqrt(np.diag(S + eps))) *  S * np.diag(1/np.sqrt(np.diag(S + eps)))
    r_hat_gamma = gamma * r_hat + (1 - gamma)*np.eye(ns)
    Sigma = np.diag(np.sqrt(np.diag(S))) * r_hat_gamma *  np.diag(np.sqrt(np.diag(S)))
    # print('Sigma', Sigma.shape)
    # print(1/0)
    return Sigma

    