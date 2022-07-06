"""This module contains helper methods used to configure BSL."""

import matplotlib.pyplot as plt
import numpy as np

import elfi.visualization.visualization as vis
from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood
from elfi.methods.utils import batch_to_arr2d
from elfi.model.utils import get_summary_names

def plot_summary_statistics(model, theta, n_samples, summary_names=None):
    """Plot simulated summary statistics at parameter values theta.

    Intent is to check distribution shape, particularly normality, for BSL inference.

    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    theta : dict or np.array
        Model parameters which are used to run the simulations.
    n_samples: int
        Number of simulations.
    summary_names : list, optional
        Summaries which are plotted. Defaults to all summary statistics in the model.

    """
    params = theta if isinstance(theta, dict) else dict(zip(theta, model.parameter_names))
    summary_names = summary_names or get_summary_names(model)
    ssx = model.generate(n_samples, outputs=summary_names, with_values=params)

    ssx_dict = {}
    for output_name in summary_names:
        if ssx[output_name].ndim > 1 and ssx[output_name].shape[1] > 1:
            ns = ssx[output_name].shape[1]
            for i in range(ns):
                new_output_name = output_name + '_' + str(i+1)
                ssx_dict[new_output_name] = ssx[output_name][:, i]
        else:
            ssx_dict[output_name] = ssx[output_name]

    vis.plot_summaries(ssx_dict)


def plot_covariance_matrix(model, theta, n_samples, summary_names=None, corr=False, 
                           precision=False, colorbar=True):
    """Plot correlation matrix of summary statistics.

    Check sparsity of covariance (or correlation) matrix.
    Useful to determine if shrinkage estimation could be applied
    which can reduce the number of model simulations required. 
    
    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    theta : dict or np.array
        Model parameters which are used to run the simulations.
    n_samples: int
        Number of simulations.
    summary_names : list, optional
        Summaries which are plotted. Defaults to all summary statistics in the model.
    corr : bool
        True -> correlation, False -> covariance
    precision: bool
        True -> precision matrix, False -> covariance/corr
    
    """
    params = theta if isinstance(theta, dict) else dict(zip(theta, model.parameter_names))
    summary_names = summary_names or get_summary_names(model)
    ssx = model.generate(n_samples, outputs=summary_names, with_values=params)
    ssx_arr = batch_to_arr2d(ssx, summary_names)

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


def log_SL_stdev(model, theta, sl_samples, M, sl_method=None, summary_names=None):
    """Estimate the standard deviation of the log SL.
    
    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    theta : dict or np.array
        Model parameters which are used to run the simulations.
    sl_samples : int
        Number of simulations used in synthetic likelihood estimation.
    M : int
        Number of log-likelihoods to estimate standard deviation.
    sl_method : callable
        Synthetic likelihood estimation method. Defaults to gaussian_syn_likelihood.
    summary_names : list
           Summaries which are plotted. Defaults to all summary statistics in the model.

    Returns
    -------
    float

    """
    params = theta if isinstance(theta, dict) else dict(zip(theta, model.parameter_names))
    sl_method = sl_method or gaussian_syn_likelihood
    summary_names = summary_names or get_summary_names(model)
    observed = np.column_stack([model[node].observed for node in summary_names])

    ll = np.zeros(M)
    for i in range(M):
        ssx = model.generate(sl_samples, outputs=summary_names, with_values=params)
        ssx_arr = batch_to_arr2d(ssx, summary_names)
        ll[i] = sl_method(ssx_arr, observed)
    return np.std(ll)
