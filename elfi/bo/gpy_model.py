import numpy as np
import GPy

class GpyModel():

    # Defaults
    kernel_class = GPy.kern.RBF
    kernel_var = 1.0
    kernel_lengthscale = 0.1
    noise_var = 1.0
    # optimizers: lbfgsb, simplex, scg, adadelta, rasmussen
    optimizer = "scg"
    opt_max_iters = int(1e3)

    def __init__(self, input_dim, kernel=None, bounds=None):
        self.input_dim = input_dim
        if self.input_dim < 1:
            raise ValueError("input_dim needs to be larger than 1")
        if bounds is not None:
            self.bounds = bounds
        else:
            print("GpyModel: No bounds supplied, defaulting to [0,1] bounds.")
            self.bounds = [(0,1)] * self.input_dim
        if len(self.bounds) != self.input_dim:
            raise ValueError("Bounds dimensionality doesn't match with input_dim")
        self.kernel = kernel or self._get_kernel()
        if self.kernel.input_dim != self.input_dim:
            raise ValueError("Kernel input_dim must match model input_dim")
        self.gp = None

    def evaluate(self, x):
        """ Returns the mean, variance and std of the GP at x as floats """
        if self.gp is None:
            return 0.0, 0.0, 0.0
        m, s2 = self.gp.predict(np.atleast_2d(x))
        return float(m), float(s2), np.sqrt(float(s2))

    def eval_mean(self, x):
        m, s2, s = self.evaluate(x)
        return m

    def _get_kernel(self):
        """ Internal function to create a kernel for GPy model
        Available at least: exponential, expquad, matern32, matern52
        """

        if isinstance(self.kernel_class, str):
            self.kernel_class = getattr(GPy.kern, self.kernel_class)
        return self.kernel_class(input_dim=self.input_dim,
                                 variance=self.kernel_var,
                                 lengthscale=self.kernel_lengthscale)

    def _get_gp(self, X, Y):
        return GPy.models.GPRegression(X=X, Y=Y,
                                       kernel=self.kernel,
                                       noise_var=self.noise_var,
                                       normalizer=True)

    def _within_bounds(self, x):
        for i, xi in enumerate(x):
            if not self.bounds[i][0] <= xi <= self.bounds[i][1]:
                return False
        return True

    def _check_input(self, X, Y):
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray) or len(X.shape) > 2 or len(Y.shape) > 2:
            raise ValueError("Observation arrays X and Y must be 2d numpy arrays (X type=%s, Y type=%s)" % (type(X), type(Y)))
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        for x in X:
            if not self._within_bounds(x):
                raise ValueError("Location %s was not within model bounds." % (x))
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Observation arrays X and Y must be of equal length (X len=%d, Y len=%d)" % (X.shape[0], Y.shape[0]))
        if X.shape[1] != self.input_dim or Y.shape[1] != 1:
            raise ValueError("Dimension of X (%d) must agree with model dimension (%d), dimension of Y (%d) must be 1." % (X.shape[1], self.input_dim, Y.shape[1]))
        return X, Y

    def update(self, X, Y):
        """
            Add (X, Y) as observations, updates GP model.
            X and Y should be 2d numpy arrays with observations in rows.
        """
        X, Y = self._check_input(X, Y)
        #print("GpyModel: Observed: %s at %s" % (X, Y))
        if self.gp is None:
            self.gp = self._get_gp(X, Y)
        else:
            X = np.vstack((self.gp.X, X))
            Y = np.vstack((self.gp.Y, Y))
            self.gp.set_XY(X, Y)
            self.gp.num_data = X.shape[0]  # bug in GPy
        old_gp = self.gp.copy()
        try:
            self.gp.optimize(self.optimizer, max_iters=self.opt_max_iters)
            del old_gp
        except np.linalg.linalg.LinAlgError as e:
            print("GpyModel: Numerical error in GP optimization! Reverting to previous model.")
            self.gp = old_gp

    def n_observations(self):
        """ Returns the number of observed samples """
        if self.gp is None:
            return 0
        return self.gp.num_data

