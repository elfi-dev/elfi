"""Implementations for ratio estiomation classifiers."""

import abc

import numpy as np


class Classifier(abc.ABC):
    """An abstract base class for a ratio estimation classifier."""

    @abc.abstractmethod
    def __init__(self):
        """Initialize a classifier."""
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit a classifier.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.
        y: np.ndarray (n_samples, )
            Target values, bust be binary.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_log_likelihood_ratio(self, X):
        """Predict a log-likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.

        Returns
        -------
        np.ndarray

        """
        raise NotImplementedError

    def predict_likelihood_ratio(self, X):
        """Predict a likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.

        Returns
        -------
        np.ndarray

        """
        return np.exp(self.predict_log_likelihood_ratio(X))

    @property
    @abc.abstractmethod
    def attributes(self):
        """Return attributes dictionary."""
        raise NotImplementedError
