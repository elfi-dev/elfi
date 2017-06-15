"""
Post-processing for posterior approximations from other ABC algorithms.

See the review paper:

Fundamentals and Recent Developments in Approximate Bayesian Computation
Lintusaari et. al
Syst Biol (2017) 66 (1): e66-e82.
https://doi.org/10.1093/sysbio/syw077

for more information.
"""

from sklearn.linear_model import LinearRegression
import numpy as np

from . import results


__all__ = ('LinearAdjustment', 'adjust_posterior')


class RegressionAdjustment(object):
    """Base class for regression adjustments.

    Each parameter is assumed to be a scalar. A local regression is
    fitted for each parameter individually using the values of the
    summary statistics as the regressors.  The regression model can be
    any object implementing a 'fit()' method. All keyword arguments
    given to the constructor are passed to the regression model.

    Subclasses need to implement the methods '_adjust' and
    '_input_variables'.  They must also specify the class variables
    '_regression_model' and '_name'.  See the individual documentation
    and the 'LinearAdjustment' class for further detail.

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
    parameters
      a generator for the values of the parameters
    result
      the result object from an ABC algorithm
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
        self._result = None
        self._parameter_names = None

    @property
    def parameter_names(self):
        self._check_fitted()
        return self._parameter_names

    @property
    def parameters(self):
        """Iterator for parameter values"""
        self._check_fitted()
        
        def iterator():
            for name in self.parameter_names:
                yield self.result.outputs[name]

        return iterator()

    @property
    def result(self):
        """Result object"""
        self._check_fitted()
        return self._result

    @property
    def X(self):
        """The input variables"""
        self._check_fitted()
        return self._X

    def _check_fitted(self):
        if not self._fitted:
            raise ValueError("The regression model must be fitted first. Use the fit() method.")

    def fit(self, model, result, parameter_names, summary_names):
        """Fit a regression adjustment model to the posterior result.

        Parameters
        ----------
        model : elfi.ElfiModel
          the inference model
        result : elfi.methods.Result
          a result object from an ABC method
        parameter_names : list[str]
          a list of parameter names
        summary_names : list[str]
          a list of names for the summary nodes
        """
        self._X = self._input_variables(model, result, summary_names)
        self._result = result
        self._parameter_names = parameter_names

        for name in parameter_names:
            self.regression_models.append(self._fit1(self._X, result.outputs[name]))

        self._fitted = True

    def _fit1(self, X, y):
        return self._regression_model(**self._model_kwargs).fit(X, y)

    def adjust(self):
        """Adjust the posterior.

        Returns
        -------
          a Result object containing the adjusted posterior
        """
        outputs = {}
        for (i, name) in enumerate(self.parameter_names):
            theta_i = self.result.outputs[name]
            adjusted = self._adjust(theta_i, self.regression_models[i])
            outputs[name] = adjusted

        res = results.Result(method_name=self._name, outputs=outputs,
                             parameter_names=self._parameter_names)
        return res

    def _adjust(self, theta_i, regression_model):
        """Adjust a single parameter using a fitted regression model.

        Parameters
        ----------
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

    def _input_variables(self, model, result, summary_names):
        """Construct a matrix of regressors.

        Parameters
        ----------
        model : elfi.ElfiModel
          the inference model
        result
          a result object from an ABC algorithm
        summary_names : list[str]
          names of the summary nodes

        Returns
        -------
        X
          a numpy array of regressors
        """
        raise NotImplementedError


class LinearAdjustment(RegressionAdjustment):
    """Regression adjustment using a local linear model."""
    _regression_model = LinearRegression
    _name = 'LinearAdjustment'

    def __init__(self, **kwargs):
        super(LinearAdjustment, self).__init__(**kwargs)

    def _adjust(self, theta_i, regression_model):
        b = regression_model.coef_
        return theta_i - self.X.dot(b)
        
    def _input_variables(self, model, result, summary_names):
        """Regress on the differences to the observed summaries."""
        observed_summaries = np.stack([model[s].observed for s in summary_names], axis=1)
        summaries = np.stack([result.outputs[name] for name in summary_names], axis=1)
        return summaries - observed_summaries


def adjust_posterior(model, result, parameter_names, summary_names, adjustment=None):
    """Adjust the posterior using local regression.

    Note that the summary nodes need to be explicitly included to the
    result object with the 'outputs' keyword argument when performing
    the inference.

    Parameters
    ----------
    model : elfi.ElfiModel
      the inference model
    result : elfi.
      a result object from an ABC algorithm
    parameter_names : list[str]
      names of the parameters
    summary_names : list[str]
      names of the summary nodes
    adjustment : RegressionAdjustment
      a regression adjustment object

    Returns
    -------
    result
      a Result object with the adjusted posterior

    Examples
    --------

    >>> import elfi
    >>> from elfi.examples import bdm
    >>> m = bdm.get_model()
    >>> res = elfi.Rejection(m['d'], outputs=['T1']).sample(1000)
    >>> adj = adjust_posterior(m, res, ['alpha'], ['T1'], LinearAdjustment())
    """
    adjustment = adjustment or LinearAdjustment()
    adjustment.fit(model, result, parameter_names, summary_names)
    return adjustment.adjust()
