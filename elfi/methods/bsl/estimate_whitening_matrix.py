import numpy as np
from scipy import linalg
from scipy.stats import norm
import elfi

def estimate_whitening_matrix(model, summary_names, theta_point, batch_size=1,
                              method="bsl", *args, **kwargs):
    #TODO: Idea -> generate ssx from model + output_names, zip params

    # batch_summaries = model.generate(batch_size, summary_names)
    # print('batch_summaries', batch_summaries)
    # summary_dims = 0
    # for summary_key in batch_summaries:
    #     summary_dims += len(batch_summaries[summary_key][0])
    # print('summary_dims', summary_dims)
    # ssx = np.zeros((batch_size, summary_dims))
        
    # for i in range(batch_size):
    #     batch_iter = []
    #     for summary_key in batch_summaries:
    #         batch_iter = np.append(batch_iter, batch_summaries[summary_key][i])
    #     ssx[i, :] = batch_iter

    # print('ssx', ssx)
    
    # print('batch_summaries', np.column_stack(batch_summaries)
    m = model.copy()
    bsl_temp = elfi.BSL(m, summary_names=summary_names, method=method,
            batch_size=batch_size, chains=1, chain_length=1, burn_in=0,
            )
    ssx = bsl_temp.estimate_whitening_matrix_helper(theta_point)

    ns, n = ssx.shape[0:2] # get only first 2 dims
    ssx = ssx.reshape((ns, n))

    mu = np.mean(ssx, axis=0) # TODO: Assumes ssx dims (handling in init)
    std = np.std(ssx, axis=0)
    mu_mat = np.tile(np.array([mu]), (ns, 1))
    std_mat = np.tile(np.array([std]), (ns, 1))
    ssx_std = (ssx - mu_mat) / std_mat
    cov_mat = np.cov(np.transpose(ssx_std)) # TODO: Assumes ssx dims
    w, v = linalg.eig(cov_mat)
    diag_w = np.diag(np.power(w, -0.5)).real.round(8)
    w_pca = np.dot(diag_w, v.T).real.round(8)
    return w_pca