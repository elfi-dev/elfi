"""Function run prior to sampling to find whitening matrix."""

import numpy as np
import scipy.stats as ss
from scipy import linalg

import elfi
from elfi.methods.utils import batch_to_arr2d
from elfi.model.elfi_model import ElfiModel, Summary


def estimate_whitening_matrix(model, batch_size, theta_point, method="bsl",
                              summary_names=None, seed=None):
    """Estimate the whitening matrix to be used in wBsl and wsemiBsl methods.

    Details are outlined in Priddle et al. 2021.

    References
    ----------
    Jacob W. Priddle, Scott A. Sisson, David T. Frazier, Ian Turner &
    Christopher Drovandi (2021)
    Efficient Bayesian Synthetic Likelihood with Whitening Transformations,
    Journal of Computational and Graphical Statistics,
    DOI: 10.1080/10618600.2021.1979012

    Parameters
    ----------
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm
    batch_size: int
        Number of simulations.
    theta_point: array-like
        Array-like value for theta thought to be close to true value.
        The simulated summaries are found at this point.
    method : str, optional
        Method for which the whitening matrix is estimated, "bsl" (default) or "semibsl".
    summary_names : str or list, optional
        Summaries used in synthetic likelihood estimation. Defaults to all summary statistics.

    Returns
    -------
    W: np.array of shape (N, N)
        Whitening matrix used to decorrelate the simulated summaries.

    """
    if summary_names is None:
        summary_names = [node for node in model.nodes if isinstance(model[node], Summary)
                         and not node.startswith('_')]
    if isinstance(summary_names, str):
        summary_names = [summary_names]
    param_values = dict(zip(model.parameter_names, theta_point))
    ssx = model.generate(batch_size, outputs=summary_names, with_values=param_values, seed=seed)
    ssx = batch_to_arr2d(ssx, summary_names)
    ns, n = ssx.shape

    if method == "semibsl":
        sim_eta = np.zeros(ssx.shape)
        for j in range(ssx.shape[1]):
            ssx_j = ssx[:, j]
            sim_eta[:, j] = ss.norm.ppf(ss.rankdata(ssx_j)/(ssx.shape[0]+1))
        ssx = sim_eta

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
