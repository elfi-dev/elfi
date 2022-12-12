"""Functions to calculate covariance and correlation with Warton shrinkage."""

import numpy as np


def cov_warton(S, gamma):
    """Apply warton shrinkage to sample covariance matrix.

    Parameters
    ----------
    S: np.ndarray
        2D array for sample covariance of simulated summaries
    gamma : np.float64
        shrinkage parameter
    Returns
    -------
    Sigma: np.ndarray
        2D array for ridge estimator of the covariance matrix

    """
    if gamma < 0 or gamma > 1:
        raise ValueError("Gamma must be between 0 and 1")
    _, ns = S.shape
    eps = 1e-5  # prevent divide by 0 for r_hat
    D1 = np.diag(1/np.sqrt(np.diag(S + eps)))
    D2 = np.diag(np.sqrt(np.diag(S + eps)))
    r_hat = np.matmul(np.matmul(D1, S), D1)
    r_hat_gamma = gamma * r_hat + (1 - gamma)*np.eye(ns)
    Sigma = np.matmul(np.matmul(D2, r_hat_gamma), D2)
    return Sigma


def corr_warton(R, gamma):
    """Apply warton shrinkage to sample correlation matrix.

    Parameters
    ----------
    R: np.ndarray
        2D array for sample covariance of simulated summaries
    gamma : np.float64
        shrinkage parameter

    Returns
    -------
    Sigma: np.ndarray
        2D array for ridge estimator of the covariance matrix

    """
    _, ns = R.shape
    return gamma * R + (1 - gamma) * np.eye(ns)
