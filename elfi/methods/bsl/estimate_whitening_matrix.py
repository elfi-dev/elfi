"""Function run prior to sampling to find whitening matrix."""

import numpy as np
import scipy.stats as ss
from scipy import linalg

from elfi.methods.utils import batch_to_arr2d


def estimate_whitening_matrix(model, n_sim, theta, feature_names, likelihood_type="standard",
                              seed=None):
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
    n_sim: int
        Number of simulations.
    theta: dict or array-like
        Parameter values thought to be close to true value.
        The simulated features are found at this point.
    feature_names : str or list
        Features used in synthetic likelihood estimation.
    likelihood_type : str, optional
        Synthetic likelihood type, "standard" (default) or "semiparametric".
    seed : int, optional
        Seed for data generation.

    Returns
    -------
    W: np.array of shape (N, N)
        Whitening matrix used to decorrelate the simulated features.

    """
    if likelihood_type not in ["standard", "semiparametric"]:
        raise ValueError("Unsupported likelihood type \'{}\'.".format(likelihood_type))

    param_values = theta if isinstance(theta, dict) else dict(zip(model.parameter_names, theta))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names

    ssx = model.generate(n_sim, outputs=feature_names, with_values=param_values, seed=seed)
    ssx = batch_to_arr2d(ssx, feature_names)
    ns, n = ssx.shape

    if likelihood_type == "semiparametric":
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
