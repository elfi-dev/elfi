"""This module contains methods for density ratio estimation."""

import logging
from functools import partial

import numpy as np

logger = logging.getLogger(__name__)


def calculate_densratio_basis_sigma(sigma_1, sigma_2):
    """Heuristic way to choose a basis sigma for density ratio estimation.

    Parameters
    ----------
    sigma_1 : float
        Standard deviation related to population 1
    sigma_2 : float
        Standard deviation related to population 2

    Returns
    -------
    float
        Basis function scale parameter that works often well in practice.

    """
    sigma = sigma_1 * sigma_2 / np.sqrt(np.abs(sigma_1 ** 2 - sigma_2 ** 2))
    return sigma


class DensityRatioEstimation:
    """A density ratio estimation class."""

    def __init__(self,
                 n=100,
                 epsilon=0.1,
                 max_iter=500,
                 abs_tol=0.01,
                 conv_check_interval=20,
                 fold=5,
                 optimize=False):
        """Construct the density ratio estimation algorithm object.

        Parameters
        ----------
        n : int
            Number of RBF basis functions.
        epsilon : float
            Parameter determining speed of gradient descent.
        max_iter : int
            Maximum number of iterations used in gradient descent optimization of the weights.
        abs_tol : float
            Absolute tolerance value for determining convergence of optimization of the weights.
        conv_check_interval : int
            Integer defining the interval of convergence checks in gradient descent.
        fold : int
            Number of folds in likelihood cross validation used to optimize basis scale-params.
        optimize : boolean
            Boolean indicating whether or not to optimize RBF scale.

        """
        self.n = n
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.fold = fold
        self.sigma = None
        self.conv_check_interval = conv_check_interval
        self.optimize = optimize

    def fit(self,
            x,
            y,
            weights_x=None,
            weights_y=None,
            sigma=None):
        """Fit the density ratio estimation object.

        Parameters
        ----------
        x : array
            Sample from the nominator distribution.
        y : sample
            Sample from the denominator distribution.
        weights_x : array
            Vector of non-negative nominator sample weights, must be able to normalize.
        weights_y : array
            Vector of non-negative denominator sample weights, must be able to normalize.
        sigma : float or list
            List of RBF kernel scales, fit selected at initial call.

        """
        self.x_len = x.shape[0]
        self.y_len = y.shape[0]
        x = x.reshape(self.x_len, -1)
        y = y.reshape(self.y_len, -1)
        self.x = x

        if self.x_len < self.n:
            raise ValueError("Number of RBFs ({}) can't be larger "
                             "than number of samples ({}).".format(self.n, self.x_len))

        self.theta = x[:self.n, :]
        if weights_x is None:
            weights_x = np.ones(self.x_len)
        if weights_y is None:
            weights_y = np.ones(self.y_len)

        self.weights_x = weights_x / np.sum(weights_x)
        self.weights_y = weights_y / np.sum(weights_y)

        self.x0 = np.average(x, axis=0, weights=weights_x)

        if isinstance(sigma, float):
            self.sigma = sigma
            self.optimize = False
        if self.optimize:
            if isinstance(sigma, list):
                scores_tuple = zip(*[self._KLIEP_lcv(x, y, sigma_i)
                                   for sigma_i in sigma])

                self.sigma = sigma[np.argmax(scores_tuple)]
            else:
                raise ValueError("To optimize RBF scale, "
                                 "you need to provide a list of candidate scales.")

        if self.sigma is None:
            raise ValueError("RBF width (sigma) has to provided in first call.")

        A = self._compute_A(x, self.sigma)
        b, b_normalized = self._compute_b(y, self.sigma)

        alpha = self._KLIEP(A, b, b_normalized, weights_x, self.sigma)
        self.w = partial(self._weighted_basis_sum, sigma=self.sigma, alpha=alpha)

    def _gaussian_basis(self, x, x0, sigma):
        """N-D RBF basis-function with equal scale-parameter for every dim."""
        return np.exp(-0.5 * np.sum((x - x0) ** 2) / sigma / sigma)

    def _weighted_basis_sum(self, x, sigma, alpha):
        """Weighted sum of gaussian basis functions evaluated at x."""
        return np.dot(np.array([[self._gaussian_basis(j, i, sigma) for j in self.theta]
                                for i in np.atleast_2d(x)]), alpha)

    def _compute_A(self, x, sigma):
        A = np.array([[self._gaussian_basis(i, j, sigma) for j in self.theta] for i in x])
        return A

    def _compute_b(self, y, sigma):
        b = np.sum(np.array(
                [[self._gaussian_basis(i, y[j, :], sigma) * self.weights_y[j]
                  for j in np.arange(self.y_len)]
                 for i in self.theta]), axis=1)
        b_normalized = b / np.dot(b.T, b)
        return b, b_normalized

    def _KLIEP_lcv(self, x, y, sigma):
        """Compute KLIEP scores for fold-folds."""
        A = self._compute_A(x, sigma)
        b, b_normalized = self._compute_b(y, sigma)

        non_null = np.any(A > 1e-64, axis=1)
        non_null_length = sum(non_null)
        if non_null_length == 0:
            return np.Inf

        A_full = A[non_null, :]
        x_full = x[non_null, :]
        weights_x_full = self.weights_x[non_null]

        fold_indices = np.array_split(np.arange(non_null_length), self.fold)
        score = np.zeros(self.fold)
        for i_fold, fold_index in enumerate(fold_indices):
            fold_index_minus = np.setdiff1d(np.arange(non_null_length), fold_index)
            alpha = self._KLIEP(A=A_full[fold_index_minus, :], b=b, b_normalized=b_normalized,
                                weights_x=weights_x_full[fold_index_minus], sigma=sigma)
            score[i_fold] = np.average(
                np.log(self._weighted_basis_sum(x_full[fold_index, :], sigma, alpha)),
                weights=weights_x_full[fold_index])

        return [np.mean(score)]

    def _KLIEP(self, A, b, b_normalized, weights_x, sigma):
        """Kullback-Leibler Importance Estimation Procedure using gradient descent."""
        alpha = 1 / self.n * np.ones(self.n)
        target_fun_prev = self._weighted_basis_sum(x=self.x, sigma=sigma, alpha=alpha)
        abs_diff = 0.0
        non_null = np.any(A > 1e-64, axis=1)
        A_full = A[non_null, :]
        weights_x_full = weights_x[non_null]
        for i in np.arange(self.max_iter):
            dAdalpha = np.matmul(A_full.T, (weights_x_full / (np.matmul(A_full, alpha))))
            alpha += self.epsilon * dAdalpha
            alpha = np.maximum(0, alpha + (1 - np.dot(b.T, alpha)) * b_normalized)
            alpha = alpha / np.dot(b.T, alpha)
            if np.remainder(i, self.conv_check_interval) == 0:
                target_fun = self._weighted_basis_sum(x=self.x, sigma=sigma, alpha=alpha)
                abs_diff = np.linalg.norm(target_fun - target_fun_prev)
                if abs_diff < self.abs_tol:
                    break
                target_fun_prev = target_fun
        return alpha

    def max_ratio(self):
        """Find the maximum of the density ratio at numerator sample."""
        max_value = np.max(self.w(self.x))
        return max_value
