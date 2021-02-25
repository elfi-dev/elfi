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
    # print('paramlen', len(param))
    P = np.diag(np.zeros(n_rows))
    # print('PPPP', P)
    # print('np.tril_indices(n_rows, -1)', np.triu_indices(n_rows, 1))
    P[np.triu_indices(n_rows, 1)] = param
    # print('param', param)
    P = np.add(P, np.transpose(P))
    np.fill_diagonal(P, 1)
#     print('PPP', P)
    return P


def gaussian_copula_density(rho_hat, u, sd):
    eta = norm.ppf(u)
    dim = len(u)
    # print('dimdim', dim)
    # print('u', u, 'eta', eta)
    # print('eta', eta.shape)
    # print('np.linalg.det(rho)', np.linalg.det(rho))
    # print('np.linalg.inv(rho)', np.linalg.inv(rho))
    # print('np.subtract(np.linalg.inv(rho), np.eye(dim))', np.subtract(np.linalg.inv(rho), np.eye(dim)))
    # print('(-1/2)*np.transpose(eta)', np.array([(-1/2)*np.transpose(eta)])
    # print('np.subtract(np.linalg.inv(rho), np.eye(dim)) * eta', np.subtract(np.linalg.inv(rho), np.eye(dim)) * eta)
    eta = np.array(eta).reshape(dim, 1)
    eta[np.isposinf(eta)] = 1e+10 # TODO: CHECK WHY THESE CHECKS NEEDED
    eta[np.isneginf(eta)] = -1e+10
    # print('uuuu', u)
    # print('etaeta', eta)
    # print('etashape', eta.shape)
    rho = p2P(rho_hat, dim)
    # print('rho', rho)

    try:
        test = np.linalg.cholesky(rho)
    except Exception:
        raise("rho not SPD")
    # part1 =  np.array([(-1/2)*np.transpose(eta)])
    # part2 =  np.subtract(np.linalg.inv(rho), np.eye(dim)) * eta

    # print('part1', part1.shape)
    # print('part2', part2.shape)
    # combined = part1 * part2
    # print('combined', combined.shape)



#     res = 1/np.sqrt((np.linalg.det(rho))) * \
#           np.exp(np.matmul(np.array([(-1/2)*np.transpose(eta)]),
#           np.matmul(np.subtract(np.linalg.inv(rho), np.eye(dim)), eta)))
    # print('np.linalg.inv(rho)', np.linalg.inv(rho))
    det = np.linalg.det(rho)
    # print('detdet', det)
    sign, logdet = np.linalg.slogdet(rho)
    mat = np.subtract(np.linalg.inv(rho), np.eye(dim))
    # print('matmat', mat)
    mat_res = np.dot(np.dot(np.transpose(eta), mat), eta)


    # print('mat_res', mat_res)
    # print('logdet', logdet)
    # print(1/0)
    # print('sig/n', sign)
    res = - (( logdet + mat_res) / 2)



    # distval = mahalanobis(eta, np.zeros(dim), np.linalg.inv(rho) - np.eye(dim)) ** 2
    
    # comp_dist_val = 

    # logretval = -(logdet + distval) /2 # removed dim * log(2pi)

    # print('logretval', logretval)

    # print('distval', distval)

    # print('logdet', logdet)
    # print('np.linalg.inv(rho)', np.linalg.inv(rho))
    # print('comp2', np.matmul(np.transpose(eta),
    # np.matmul(np.subtract(np.linalg.inv(rho), np.eye(dim)), eta)

    # print('eta', eta.shape)

    # sd_mat = np.diag(sd)
    # cov_mat = sd_mat * rho * sd_mat
    # mvn_res = multivariate_normal.logpdf(np.transpose(eta), mean=np.zeros(len(eta)), cov=(rho - np.eye(dim)) )
    # print('mvn_res', mvn_res)
    # print('resresintheres', res)
#     print('res_log', np.log(res[0]))

#     print(1/0)
#     return np.log(res[0])
    return res