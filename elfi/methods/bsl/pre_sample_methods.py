"""This module contains methods that assist with setting up synthetic likelihood calculation."""

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy import linalg
from sklearn.exceptions import ConvergenceWarning

import elfi.visualization.visualization as vis
from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood
from elfi.methods.utils import batch_to_arr2d

logger = logging.getLogger(__name__)


def plot_features(model, theta, n_sim, feature_names, seed=None):
    """Plot simulated feature values at theta.

    Intent is to check distribution shape, particularly normality, for BSL inference.

    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    theta : dict or np.array
        Model parameters which are used to run the simulations.
    n_sim: int
        Number of simulations.
    feature_names : list or str
        Features which are plotted.
    seed : int, optional
        Seed for data generation.

    """
    params = theta if isinstance(theta, dict) else dict(zip(model.parameter_names, theta))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    ssx = model.generate(n_sim, outputs=feature_names, with_values=params, seed=seed)

    ssx_dict = {}
    for output_name in feature_names:
        if ssx[output_name].ndim > 1 and ssx[output_name].shape[1] > 1:
            ns = ssx[output_name].shape[1]
            for i in range(ns):
                new_output_name = output_name + '_' + str(i+1)
                ssx_dict[new_output_name] = ssx[output_name][:, i]
        else:
            ssx_dict[output_name] = ssx[output_name]

    vis.plot_marginals(ssx_dict, ncols=int(np.ceil(np.sqrt(len(ssx_dict)))), bins=30)


def plot_covariance_matrix(model, theta, n_sim, feature_names, corr=False,
                           precision=False, colorbar=True, seed=None):
    """Plot correlation matrix of simulated features.

    Check sparsity of covariance (or correlation) matrix.
    Useful to determine if shrinkage estimation could be applied
    which can reduce the number of model simulations required.

    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    theta : dict or np.array
        Model parameters which are used to run the simulations.
    n_sim : int
        Number of simulations.
    feature_names : list or str
        Features which are plotted.
    corr : bool, optional
        True -> correlation, False -> covariance
    precision: bool, optional
        True -> precision matrix, False -> covariance/corr
    colorbar : bool, optional
        Whether to include colorbar in the plot.
    seed : int, optional
        Seed for data generation.

    """
    params = theta if isinstance(theta, dict) else dict(zip(model.parameter_names, theta))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    ssx = model.generate(n_sim, outputs=feature_names, with_values=params, seed=seed)
    ssx_arr = batch_to_arr2d(ssx, feature_names)

    sample_cov = np.cov(ssx_arr, rowvar=False)
    if corr:
        sample_cov = np.corrcoef(sample_cov)  # correlation matrix
    if precision:
        sample_cov = np.linalg.inv(sample_cov)

    fig = plt.figure()
    ax = plt.subplot(111)

    cax = ax.matshow(sample_cov)
    if colorbar:
        fig.colorbar(cax)


def log_SL_stdev(model, theta, n_sim, feature_names, likelihood=None, M=20, seed=None):
    """Estimate the standard deviation of the log synthetic likelihood.

    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    theta : dict or np.array
        Model parameters which are used to run the simulations.
    n_sim : int or array_like
        Number of simulations used to calculate the synthetic likelihood estimates.
    feature_names : list or str
        Features used in synthetic likelihood estimation.
    likelihood : callable, optional
        Synthetic likelihood estimation method. Defaults to gaussian_syn_likelihood.
    M : int, optional
        Number of log-likelihoods to estimate standard deviation.
    seed : int, optional
        Seed for data generation.

    Returns
    -------
    np.array

    """
    params = theta if isinstance(theta, dict) else dict(zip(model.parameter_names, theta))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    observed = np.column_stack([model[node].observed for node in feature_names])
    likelihood = likelihood or gaussian_syn_likelihood

    n_sim = np.atleast_1d(n_sim)
    max_sim = max(n_sim)
    ll = np.zeros((len(n_sim), M))

    child_seeds = np.random.SeedSequence(seed).generate_state(M)
    for i in range(M):
        seed_i = child_seeds[i]
        ssx = model.generate(max_sim, outputs=feature_names, with_values=params, seed=seed_i)
        ssx_arr = batch_to_arr2d(ssx, feature_names)
        for n_i, n in enumerate(n_sim):
            ll[n_i, i] = likelihood(ssx_arr[:n], observed)
    return np.std(ll, axis=1)


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


def select_penalty(model, n_sim, theta, feature_names, likelihood=None,
                   lmdas=None, M=20, sigma=1.5, shrinkage="glasso",
                   whitening=None, seed=None, verbose=False):
    """Select the penalty value to use within an MCMC BSL algorithm.

    Selects the penalty (lambda) value that gives the closest estimated
    loglik standard deviation closest to sigma for each specified
    batch_size.

    Parameters
    ----------
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm
    n_sim : int or np.array
        The number of simulations. If array, selects penalty for each simulation count.
    theta : dict or np.array
        Parameter point where all loglikelihoods are calculated.
    feature_names : str or list
        Features used in synthetic likelihood estimation.
    likelihood : callable, optional
        Synthetic likelihood estimation method. Defaults to gaussian_syn_likelihood.
    lmdas : np.array, optional
        The penalties values to test over
    M : int, optional
        The number of repeats at the same lambda and n_sim values
        to estimate the stdev of the log-likelihood
    sigma : float
        A given standard deviation value (should be between 1 and 2)
        where the lambda value with the closest estimated loglik stdev
        to sigma is returned.
    shrinkage : str, optional
        The shrinkage method to be used with the penalty param.
    whitening : np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller simulation count).
    seed : int, optional
        Seed for the data generation from the ElfiModel
    verbose : bool, optional
        Option to display additional information on stdevs

    Returns
    -------
        The closest lambdas and standard deviation values (for each batch_size passed in)

    """
    param_values = theta if isinstance(theta, dict) else dict(zip(model.parameter_names, theta))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    ssy = np.column_stack([model[node].observed for node in feature_names])

    likelihood = likelihood or gaussian_syn_likelihood

    if lmdas is None:
        if shrinkage == "glasso":
            lmdas = list(np.exp(np.arange(-5.5, -1.5, 0.2)))
        if shrinkage == "warton":
            lmdas = list((np.arange(0.2, 0.8, 0.02)))

    n_lambda = len(lmdas)
    batch_size = np.array([n_sim]).flatten()
    ns = len(batch_size)

    child_seeds = np.random.SeedSequence(seed).generate_state(M)

    logliks = np.zeros((M, ns, n_lambda))

    with warnings.catch_warnings():
        # ignore graphical lasso bad values
        warnings.simplefilter('ignore', category=ConvergenceWarning)

        for m_iteration in range(M):  # for M logliks at same penalty and batch_size
            ssx = model.generate(max(batch_size),
                                 outputs=feature_names,
                                 with_values=param_values,
                                 seed=child_seeds[m_iteration])
            ssx_arr = batch_to_arr2d(ssx, feature_names)
            for n_iteration in range(ns):
                ssx_n = ssx_arr[:batch_size[n_iteration]]
                for lmda_iteration in range(n_lambda):
                    try:
                        loglik = likelihood(ssx_n,
                                            ssy,
                                            shrinkage=shrinkage,
                                            penalty=lmdas[lmda_iteration],
                                            whitening=whitening)
                    except FloatingPointError as err:
                        logger.warning('Floating point error: {}'.format(err))
                        loglik = np.NINF
                    logliks[m_iteration, n_iteration, lmda_iteration] = loglik

    # choose the lambda with the empirical s.d. of the log SL estimates
    # closest to sigma
    closest_lmdas = np.zeros(ns)
    closest_std_devs = np.zeros(ns)
    for i in range(ns):
        std_devs = np.array([np.std(logliks[:, i, j]) for j in range(n_lambda)])
        closest_arg = np.argmin(np.abs(std_devs - sigma))
        closest_lmdas[i] = lmdas[closest_arg]
        closest_std_devs[i] = std_devs[closest_arg]
    if verbose:
        print('logliks: ', logliks)
        print('std_devs: ', std_devs)
    return closest_lmdas, closest_std_devs
