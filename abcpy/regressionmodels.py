import numpy as np

class RegressionModelBase():
    """ All regression models are assumed to fulfill this interface """

    def __init__(self, n_var=0, bounds=None):
        if n_var < 1:
            raise ValueError("Number of variables needs to be larger than 1")
        self.n_var = n_var
        self.bounds = bounds or [(0,1)] * self.n_var
        if len(bounds) != self.n_var:
            raise ValueError("Number of variables needs to equal the number of bounds")
        self.Xobs = None # observation location in parameter space
        self.Yobs = None # observed values

    def evaluate(self, x):
        """
            Evaluates the model function value at 'x'
            type(x) = numpy.array

            returns: mean, variance, standard deviation
        """
        return None, None, None

    def update(self, X, Y):
        """
            Add (X, Y) as observations, updates GP model.
            Assumes X and Y are 2d numpy arrays with observations in rows.
        """
        if len(X.shape) != 2 or len(Y.shape) != 2:
            raise ValueError("Observation arrays X and Y must be 2d numpy arrays")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Observation arrays X and Y must be of equal length (X = %d, Y = %d)" % (X.shape[1], Y.shape[1]))
        if X.shape[1] != self.n_var or Y.shape[1] != 1:
            raise ValueError("Dimension of X must agree with model dimensions, dimension of Y must be 1.")
        if self.Xobs is None:
            self.Xobs = X
            self.Yobs = Y
        else:
            self.Xobs = np.vstack((self.Xobs, X))
            self.Yobs = np.vstack((self.Yobs, Y))
        self._update()


    def _update(self):
        """ Updates model based on current observations """
        pass

    def n_observations(self):
        """ Returns the number of observed samples """
        if self.Xobs is None:
            return 0
        return self.Xobs.shape[0]
