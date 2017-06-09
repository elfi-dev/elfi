"""
Post-processing
"""

import numpy as np
from sklearn.linear_model import LinearRegression


class RegressionAdjustment(object):
    def __init__(self, **kwargs):
        self._model_kwargs = kwargs
        self._fitted = False
        self.regression_models =[]

    @property
    def parameter_names(self):
        self._check_fitted()
        return self._parameter_names

    @property
    def parameters(self):
        self._check_fitted()
        
        def iterator():
            for name in self.parameter_names:
                yield self.result.outputs[name]

        return iterator()

    @property
    def result(self):
        self._check_fitted()
        return self._result

    @property
    def X(self):
        self._check_fitted()
        return self._X

    def _check_fitted(self):
        if not self._fitted:
            raise ValueError("The regression model must be fitted first. Use the fit() method.")

    def fit(self, result, summary_names, observed_summaries, parameter_names):
        self._X = _input_variables(result, observed_summaries, summary_names)
        self._result = result
        self._parameter_names = parameter_names
        for r in _response(result, parameter_names):
            self.regression_models.append(self._fit1(self._X, r))

        self._fitted = True

    def _fit1(self, X, y):
        return self._regression_model(**self._model_kwargs).fit(X, y)

    def adjust(self):
        adjusted_theta = []
        for (i, theta_i) in enumerate(self.parameters):
            adjusted_theta.append(self._adjust1(theta_i, self.regression_models[i]))

        return np.stack(adjusted_theta, axis=1)

    def _adjust1(self, theta_i, regression_model):
        raise NotImplementedError


class LinearAdjustment(RegressionAdjustment):
    _regression_model = LinearRegression

    def __init__(self, **kwargs):
        super(LinearAdjustment, self).__init__(**kwargs)

    def _adjust1(self, theta_i, regression_model):
        b = regression_model.coef_
        return theta_i - self.X.dot(b)
        

def _input_variables(result, observed_summaries, summary_names):
    """Construct a matrix of input variables from summaries."""
    summaries = np.stack([result.outputs[name] for name in summary_names], axis=1)
    return summaries - observed_summaries


def _response(result, parameter_names):
    for name in parameter_names:
        yield result.outputs[name]

