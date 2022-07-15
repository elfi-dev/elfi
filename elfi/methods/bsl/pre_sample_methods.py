"""This module contains methods for assessing selected features and simulation count."""

import matplotlib.pyplot as plt
import numpy as np

import elfi.visualization.visualization as vis
from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood
from elfi.methods.utils import batch_to_arr2d


def plot_features(model, theta, n_sim, feature_names):
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

    """
    params = theta if isinstance(theta, dict) else dict(zip(theta, model.parameter_names))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    ssx = model.generate(n_sim, outputs=feature_names, with_values=params)

    ssx_dict = {}
    for output_name in feature_names:
        if ssx[output_name].ndim > 1 and ssx[output_name].shape[1] > 1:
            ns = ssx[output_name].shape[1]
            for i in range(ns):
                new_output_name = output_name + '_' + str(i+1)
                ssx_dict[new_output_name] = ssx[output_name][:, i]
        else:
            ssx_dict[output_name] = ssx[output_name]

    vis.plot_summaries(ssx_dict)


def plot_covariance_matrix(model, theta, n_sim, feature_names, corr=False,
                           precision=False, colorbar=True):
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

    """
    params = theta if isinstance(theta, dict) else dict(zip(theta, model.parameter_names))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    ssx = model.generate(n_sim, outputs=feature_names, with_values=params)
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


def log_SL_stdev(model, theta, n_sim, feature_names, likelihood=None, M=20):
    """Estimate the standard deviation of the log SL.

    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    theta : dict or np.array
        Model parameters which are used to run the simulations.
    n_sim : int
        Number of simulations used to calculate a synthetic likelihood estimate.
    feature_names : list or str
        Features used in synthetic likelihood estimation.
    likelihood : callable, optional
        Synthetic likelihood estimation method. Defaults to gaussian_syn_likelihood.
    M : int, optional
        Number of log-likelihoods to estimate standard deviation.

    Returns
    -------
    float

    """
    params = theta if isinstance(theta, dict) else dict(zip(theta, model.parameter_names))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    observed = np.column_stack([model[node].observed for node in feature_names])
    likelihood = likelihood or gaussian_syn_likelihood

    ll = np.zeros(M)
    for i in range(M):
        ssx = model.generate(n_sim, outputs=feature_names, with_values=params)
        ssx_arr = batch_to_arr2d(ssx, feature_names)
        ll[i] = likelihood(ssx_arr, observed)
    return np.std(ll)
