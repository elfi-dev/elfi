"""Function run prior to sampling to find whitening matrix"""

import numpy as np
from scipy import linalg
from scipy.stats import norm
import elfi


def estimate_whitening_matrix(model, summary_names, theta_point, batch_size=1,
                              discrepancy_name=None, *args, **kwargs):
    """Estimate the Whitening matrix to be used in wBsl and wsemiBsl methods.
       Details are outlined in Priddle et al. 2021.


    References
    ----------
    Priddle, J. W., Sisson, S. A., Frazier, D., and Drovandi, C. C. (2020).
    Efficient Bayesian Synthetic Likelihood with whitening transformations.
    arXiv pre-print server.  #TODO: UPDATE ONCE PUBLISHED

    Args:
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm        summary_names ([type]): [description]
    theta_point: array-like
        Array-like value for theta thought to be close to true value.
        The simulated summaries are found at this point.
    batch_size: int, optional
        The number of parameter evaluations in each pass through the ELFI graph.
        When using a vectorized simulator, using a suitably large batch_size can provide
        a significant performance boost.

    Returns:
    W: np.array of shape (N, N)
        Whitening matrix used to decorrelate the simulated summaries.
    """
    m = model.copy()
    # tmp_target = summary_names[0]  # TODO? ideally wouldn't need
    bsl_temp = elfi.BSL(m[discrepancy_name],
                        summary_names=summary_names,
                        batch_size=batch_size
                        )
    ssx = bsl_temp.get_ssx(theta_point)

    ns, n = ssx.shape[0:2]  # get only first 2 dims
    ssx = ssx.reshape((ns, n))  # 2 dims same, 3 dims "flatten" to 2d

    mu = np.mean(ssx, axis=0)
    std = np.std(ssx, axis=0)
    mu_mat = np.tile(np.array([mu]), (ns, 1))
    std_mat = np.tile(np.array([std]), (ns, 1))
    ssx_std = (ssx - mu_mat) / std_mat

    cov_mat = np.cov(np.transpose(ssx_std))

    w, v = linalg.eig(cov_mat)
    diag_w = np.diag(np.power(w, -0.5)).real.round(8)

    W = np.dot(diag_w, v.T).real.round(8)

    return W
