"""Function run prior to sampling to find whitening matrix."""

import numpy as np
from scipy import linalg

import elfi
from elfi.model.elfi_model import ElfiModel, NodeReference


def resolve_model(model, target, default_reference_class=NodeReference):
    """Resolve model to get model and target (discrepancy) name."""
    if isinstance(model, ElfiModel) and target is None:
        raise NotImplementedError(
            "Please specify the target node of the inference method")

    if isinstance(model, NodeReference):
        target = model
        model = target.model

    if isinstance(target, str):
        target = model[target]

    if not isinstance(target, default_reference_class):
        raise ValueError('Unknown target node class')

    return model, target.name


def estimate_whitening_matrix(model, theta_point, batch_size=1,
                              discrepancy_name=None, seed=None):
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
    theta_point: array-like
        Array-like value for theta thought to be close to true value.
        The simulated summaries are found at this point.
    batch_size: int, optional
        The number of parameter evaluations in each pass through the ELFI graph.
        When using a vectorized simulator, using a suitably large batch_size can provide
        a significant performance boost.

    Returns
    -------
    W: np.array of shape (N, N)
        Whitening matrix used to decorrelate the simulated summaries.

    """
    seed = seed or 123
    model, discrepancy_name = resolve_model(model, discrepancy_name)
    m = model.copy()
    bsl_temp = elfi.BSL(m[discrepancy_name],
                        batch_size=batch_size,
                        seed=seed
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
