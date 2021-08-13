import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.spatial.distance import mahalanobis
import math

#TODO: Where to place ?
def p2P(param, n_rows):
    """Construct a summetric matrix 1s on the diagonal from the given
    parameter vector (from p2P in the R copula package)

    Args:
        param ([type]): [description]

        Returns:
            [type]: [description]
    """
#     num_rows, _ = param.shape
    P = np.diag(np.zeros(n_rows))
    P[np.triu_indices(n_rows, 1)] = param
    P = np.add(P, np.transpose(P))
    np.fill_diagonal(P, 1)
    return P


def P2p(P, dim):
    return list(P[np.triu_indices(dim, 1)])


def gaussian_copula_density(rho_hat, u, sd, whitening=None):
    # print('u', u)
    eta = norm.ppf(u)
    if whitening is not None:
        eta = np.matmul(whitening, eta)  # TODO: In process checking trans, order, etc
    # eta = norm.ppf(u)  # TODO? comment out?
    dim = len(u)
    eta = np.array(eta).reshape(dim, 1)
    if any(np.isinf(eta)):
        return -math.inf

    if rho_hat.ndim == 1:
        rho = p2P(rho_hat, dim)
    else:
        rho = rho_hat
    try:
        test = np.linalg.cholesky(rho)  # TODO: do better way then running chol
    except Exception:
        return -math.inf
        # raise("rho not SPD")
    _, logdet = np.linalg.slogdet(rho)

    mat = np.subtract(np.linalg.inv(rho), np.eye(dim))
    mat_res = np.dot(np.dot(np.transpose(eta), mat), eta)
    # mat_res = np.einsum('nk,ij,kn -> n', eta, mat, eta)  # TODO? CHECK
    print('mat_res', mat_res)
    # print('mat_res', np.sum(mat_res))
    res = -0.5 * (logdet + mat_res)  # TODO? ADD SIGN IN?
    
    # todo testing
    # print("multivariate_normal", multivariate_normal.logpdf(
    #         eta.flatten(),
    #         # mean=sample_mean,
    #         cov=rho))
    # print("np.sum(norm.logpdf(eta))", np.sum(norm.logpdf(eta)))
    # test_res = multivariate_normal.logpdf(
    #         eta.flatten(),
    #         # mean=sample_mean,
    #         cov=rho) - np.sum(norm.logpdf(eta))
    print('test_res', test_res)
    # res = -((logdet + np.sum(mat_res))/2)
    # res = -0.5 * np.sum(mat_res) - 0.5 * dim * logdet  # TODO? add dim?
    # print(1/0)  #TODO: DEBUGGING
    print('res', res)
    print(1/0)  # TODO: break here while debugging
    return res[0][0]