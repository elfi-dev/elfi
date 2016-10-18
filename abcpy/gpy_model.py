import numpy as np
import GPy

class GpyModel():

    gp_kernel_type = "exponential"
    gp_kernel_var = 1.0
    gp_kernel_scale = 1.0
    gp_noise_var = 1.0
    # optimizers: lbfgsb, simplex, scg, adadelta, rasmussen
    optimizer = "lbfgsb"
    opt_max_iters = 1e5

    def __init__(self, n_var=0, bounds=None):
        self.gp = None
        if n_var < 1:
            raise ValueError("Number of variables needs to be larger than 1")
        self.n_var = n_var
        self.bounds = bounds or [(0,1)] * self.n_var
        if len(bounds) != self.n_var:
            raise ValueError("Number of variables needs to equal the number of bounds")


    def evaluate(self, x):
        """ Returns the mean, variance and std of the GP at x as floats """
        if self.gp is None:
            return 0.0, 0.0, 0.0
        y_m, y_s2 = self.gp.predict(np.atleast_2d(x))
        return float(y_m), float(y_s2), np.sqrt(float(y_s2))

    def eval_mean(self, x):
        m, s2, s = self.evaluate(x)
        return m

    def _get_kernel(self):
        """ Internal function to generate kernel for GPy model """
        if self.gp_kernel_type == "exponential":
            return GPy.kern.Exponential(input_dim=self.n_var,
                                        variance=self.gp_kernel_var,
                                        lengthscale=self.gp_kernel_scale)
        elif self.gp_kernel_type == "expquad":
            return GPy.kern.ExpQuad(input_dim=self.n_var,
                                    variance=self.gp_kernel_var,
                                    lengthscale=self.gp_kernel_scale)
        elif self.gp_kernel_type == "matern32":
            return GPy.kern.Matern32(input_dim=self.n_var,
                                     variance=self.gp_kernel_var,
                                     lengthscale=self.gp_kernel_scale)
        elif self.gp_kernel_type == "matern52":
            return GPy.kern.Matern52(input_dim=self.n_var,
                                     variance=self.gp_kernel_var,
                                     lengthscale=self.gp_kernel_scale)
        else:
            raise ValueError("Unknown GP kernel type: %s" % (self.gp_kernel_type))

    def update(self, X, Y):
        """
            Add (X, Y) as observations, updates GP model.
            X and Y should be 2d numpy arrays with observations in rows.
        """
        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray) or len(X.shape) != 2 or len(Y.shape) != 2:
            raise ValueError("Observation arrays X and Y must be 2d numpy arrays (X type=%s, Y type=%s)" % (type(X), type(Y)))
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Observation arrays X and Y must be of equal length (X len=%d, Y len=%d)" % (X.shape[0], Y.shape[0]))
        if X.shape[1] != self.n_var or Y.shape[1] != 1:
            raise ValueError("Dimension of X (%d) must agree with model dimension (%d), dimension of Y (%d) must be 1." % (X.shape[1], self.n_var, Y.shape[1]))
        print("Observed: %s at %s" % (X, Y))
        if self.gp is None:
            self.gp = GPy.models.GPRegression(X=X,
                                              Y=Y,
                                              kernel=self._get_kernel(),
                                              noise_var=self.gp_noise_var)
        else:
            X = np.vstack((self.gp.X, X))
            Y = np.vstack((self.gp.Y, Y))
            self.gp.set_XY(X, Y)
        try:
            self.gp.optimize(self.optimizer, max_iters=self.opt_max_iters)
        except np.linalg.linalg.LinAlgError as e:
            print("Numerical error in GP optimization! Let's hope everything still works.")

    def n_observations(self):
        """ Returns the number of observed samples """
        if self.gp is None:
            return 0
        return self.gp.num_data

