"""Post-processing for posterior samples from other ABC algorithms.

References
----------
Fundamentals and Recent Developments in Approximate Bayesian Computation
Lintusaari et. al
Syst Biol (2017) 66 (1): e66-e82.
https://doi.org/10.1093/sysbio/syw077

"""
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression

from . import results

__all__ = ('LinearAdjustment', 'adjust_posterior')


class RegressionAdjustment(object):
    """Base class for regression adjustments.

    Each parameter is assumed to be a scalar. A local regression is
    fitted for each parameter individually using the values of the
    summary statistics as the regressors.  The regression model can be
    any object implementing a `fit()` method. All keyword arguments
    given to the constructor are passed to the regression model.

    Subclasses need to implement the methods `_adjust` and
    `_input_variables`.  They must also specify the class variables
    `_regression_model` and `_name`.  See the individual documentation
    and the `LinearAdjustment` class for further detail.

    Parameters
    ----------
    kwargs**
      keyword arguments to pass to the regression model

    Attributes
    ----------
    regression_models
      a list of fitted regression model instances
    parameter_names
      a list of parameter names
    sample
      the sample object from an ABC algorithm
    X
      the regressors for the regression model

    """

    _regression_model = None
    _name = 'RegressionAdjustment'

    def __init__(self, **kwargs):
        self._model_kwargs = kwargs
        self._fitted = False
        self.regression_models = []
        self._X = None
        self._sample = None
        self._parameter_names = None
        self._finite = []

    @property
    def parameter_names(self):
        self._check_fitted()
        return self._parameter_names

    @property
    def sample(self):
        self._check_fitted()
        return self._sample

    @property
    def X(self):
        self._check_fitted()
        return self._X

    def _check_fitted(self):
        if not self._fitted:
            raise ValueError("The regression model must be fitted first. " "Use the fit() method.")

    def fit(self, sample, model, summary_names, parameter_names=None):
        """Fit a regression adjustment model to the posterior sample.

        Non-finite values in the summary statistics and parameters
        will be omitted.

        Parameters
        ----------
        sample : elfi.methods.Sample
          a sample object from an ABC method
        model : elfi.ElfiModel
          the inference model
        summary_names : list[str]
          a list of names for the summary nodes
        parameter_names : list[str] (optional)
          a list of parameter names

        """
        self._X = self._input_variables(model, sample, summary_names)
        self._sample = sample
        self._parameter_names = parameter_names or sample.parameter_names
        self._get_finite()

        for pair in self._pairs():
            self.regression_models.append(self._fit1(*pair))

        self._fitted = True

    def _fit1(self, X, y):
        return self._regression_model(**self._model_kwargs).fit(X, y)

    def _pairs(self):
        # TODO: Access the variables through the getters
        for (i, name) in enumerate(self._parameter_names):
            X = self._X[self._finite[i], :]
            p = self._sample.outputs[name][self._finite[i]]
            yield X, p

    def adjust(self):
        """Adjust the posterior.

        Only the non-finite values used to fit the regression model
        will be adjusted.

        Returns
        -------
          a Sample object containing the adjusted posterior

        """
        outputs = {}
        for (i, name) in enumerate(self.parameter_names):
            theta_i = self.sample.outputs[name][self._finite[i]]
            adjusted = self._adjust(i, theta_i, self.regression_models[i])
            outputs[name] = adjusted

        res = results.Sample(
            method_name=self._name, outputs=outputs, parameter_names=self._parameter_names)
        return res

    def _adjust(self, i, theta_i, regression_model):
        """Adjust a single parameter using a fitted regression model.

        Parameters
        ----------
        i : int
          the index of the parameter
        theta_i : np.ndarray
          a vector of parameter values to adjust
        regression_model
          a fitted regression model

        Returns
        -------
        adjusted_theta_i : np.ndarray
          an adjusted version of the parameter values

        """
        raise NotImplementedError

    def _input_variables(self, model, sample, summary_names):
        """Construct a matrix of regressors.

        Parameters
        ----------
        model : elfi.ElfiModel
          the inference model
        sample
          a sample object from an ABC algorithm
        summary_names : list[str]
          names of the summary nodes

        Returns
        -------
        X
          a numpy array of regressors

        """
        raise NotImplementedError

    def _get_finite(self):
        # TODO: Access the variables through the getters
        finite_inputs = np.isfinite(self._X).all(axis=1)
        finite = [
            finite_inputs & np.isfinite(self._sample.outputs[p]) for p in self._parameter_names
        ]
        all_finite = all(map(all, finite))
        self._finite = finite
        if not (all(finite_inputs) and all_finite):
            warnings.warn("Non-finite inputs and outputs will be omitted.")


class LinearAdjustment(RegressionAdjustment):
    """Regression adjustment using a local linear model."""

    _regression_model = LinearRegression
    _name = 'LinearAdjustment'

    def _adjust(self, i, theta_i, regression_model):
        b = regression_model.coef_
        return theta_i - self.X[self._finite[i], :].dot(b)

    def _input_variables(self, model, sample, summary_names):
        """Regress on the differences to the observed summaries."""
        observed_summaries = np.stack([model[s].observed for s in summary_names], axis=1)
        summaries = np.stack([sample.outputs[name] for name in summary_names], axis=1)
        return summaries - observed_summaries


def adjust_posterior(sample, model, summary_names, parameter_names=None, adjustment='linear'):
    """Adjust the posterior using local regression.

    Note that the summary nodes need to be explicitly included to the
    sample object with the `output_names` keyword argument when performing
    the inference.

    Parameters
    ----------
    sample : elfi.methods.results.Sample
      a sample object from an ABC algorithm
    model : elfi.ElfiModel
      the inference model
    summary_names : list[str]
      names of the summary nodes
    parameter_names : list[str] (optional)
      names of the parameters
    adjustment : RegressionAdjustment or string
      a regression adjustment object or a string specification

      Accepted values for the string specification:
       - 'linear'

    Returns
    -------
    elfi.methods.results.Sample
      a Sample object with the adjusted posterior

    Examples
    --------
    >>> import elfi
    >>> from elfi.examples import gauss
    >>> m = gauss.get_model()
    >>> res = elfi.Rejection(m['d'], output_names=['ss_mean', 'ss_var'],
    ...                      batch_size=10).sample(500, bar=False)
    >>> adj = adjust_posterior(res, m, ['ss_mean', 'ss_var'], ['mu'], LinearAdjustment())

    """
    adjustment = _get_adjustment(adjustment)
    adjustment.fit(
        model=model, sample=sample, parameter_names=parameter_names, summary_names=summary_names)
    return adjustment.adjust()


def _get_adjustment(adjustment):
    adjustments = {'linear': LinearAdjustment}

    if isinstance(adjustment, RegressionAdjustment):
        return adjustment
    elif isinstance(adjustment, str):
        try:
            return adjustments.get(adjustment, None)()
        except TypeError:
            raise ValueError("Could not find " "adjustment method:{}".format(adjustment))
