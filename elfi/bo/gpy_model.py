import numpy as np
import copy
import GPy

class GPyModel():
    """ Gaussian Process regression model using the GPy library implementation.

    Parameters
    ----------
    input_dim : int
        number of input dimensions
    bounds : tuple of (min, max) tuples
        Input space box constraints as a tuple of pairs, one for each input dim
        Eg: ((0, 1), (0, 2), (-2, 2))
        If not supplied, defaults to (0, 1) bounds for all dimenstions.
    kernel : GPy.kern kernel
        GPy compatible kernel function
        if not None, then the other kernel_* params are ignored
    kernel_class : GPy.kern classname
        type of kernel from GPy internal kernels
    kernel_var : float
        variance of kernel
    kernel_scale : float
        lengthscale of kernel
    noise_var : float
        observation noise variance
    """

    def __init__(self, input_dim=1, bounds=None, kernel=None, kernel_class=GPy.kern.RBF,
                 kernel_var=1.0, kernel_scale=0.1, noise_var=0.0):
        self.input_dim = input_dim
        if self.input_dim < 1:
            raise ValueError("input_dim needs to be larger than 1")
        if bounds is not None:
            self.bounds = bounds
        else:
            print("GPyModel: No bounds supplied, defaulting to [0,1] bounds.")
            self.bounds = [(0,1)] * self.input_dim
        if len(self.bounds) != self.input_dim:
            raise ValueError("Bounds dimensionality doesn't match with input_dim")
        self.noise_var = noise_var
        self.gp = None
        self.set_kernel(kernel, kernel_class, kernel_var, kernel_scale)

    def evaluate(self, x):
        """ Returns the GP model mean, variance and std at x.

        Parameters
        ----------
        x : numpy 1d array
            location to evaluate at

        Returns
        -------
        gp (mean, s2, s) at x : (float, float, float)
        """
        if self.gp is None:
            # TODO: return from GP prior
            return 0.0, 0.0, 0.0
        m, s2 = self.gp.predict(np.atleast_2d(x))
        if m != m:
            print("GPyModel: Warning: Mean evaluated to %s" % (m))
        return float(m), float(s2), np.sqrt(float(s2))

    def eval_mean(self, x):
        """ Returns the GP model mean function at x.

        Parameters
        ----------
        x : numpy 1d array
            location to evaluate at

        Returns
        -------
        gp mean value at x : float
        """
        m, s2, s = self.evaluate(x)
        return m

    def set_noise_var(self, noise_var=0.0):
        """ Change GP observation noise variance and re-fit the GP.

        Parameters
        ----------
        see constructor
        """
        self.noise_var = noise_var
        if self.gp is not None:
            # re-fit gp with new noise variance
            self.gp = self._fit_gp(self.gp.X, self.gp.Y)

    def set_kernel(self, kernel=None, kernel_class=None, kernel_var=None,
                   kernel_scale=None):
        """ Changes the GP kernel to supplied and re-fit the GP.

        Parameters
        ----------
        see constructor
        """
        if kernel is not None:
            # explicit kernel supplied
            if kernel.input_dim != self.input_dim:
                raise ValueError("Kernel input_dim must match model input_dim")
            self.kernel = kernel
        else:
            self.kernel_class = kernel_class or self.kernel_class
            self.kernel_var = kernel_var or self.kernel_var
            self.kernel_scale = kernel_scale or self.kernel_scale
            if isinstance(self.kernel_class, str):
                self.kernel_class = getattr(GPy.kern, self.kernel_class)
            self.kernel = self.kernel_class(input_dim=self.input_dim,
                                            variance=self.kernel_var,
                                            lengthscale=self.kernel_scale)
        if self.gp is not None:
            # re-fit gp with new kernel
            self.gp = self._fit_gp(self.gp.X, self.gp.Y)

    def _fit_gp(self, X, Y):
        """ Constructs the gp model and returns it """
        return GPy.models.GPRegression(X=X, Y=Y,
                                       kernel=self.kernel,
                                       noise_var=self.noise_var)

    def _within_bounds(self, x):
        """ Returns true if location x is within model bounds """
        for i, xi in enumerate(x):
            if not self.bounds[i][0] <= xi <= self.bounds[i][1]:
                return False
        return True

    def _check_input(self, X, Y):
        """ Validates if input X and Y are acceptable.
        Also casts X and Y into 2d arrays if they are 1d.
        Returns (X, Y)
        """
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
        """ Add (X, Y) as observations, updates GP model.

        Parameters
        ----------
        X : numpy 1d or 2d array
            observation locations in rows (can be 1d array if only one obs)
        Y : numpy 1d or 2d array
            observation values in rows (can be 1d array if only one obs)
        """
        X, Y = self._check_input(X, Y)
        #print("GPyModel: Observed: %s at %s" % (X, Y))
        if self.gp is not None:
            X = np.vstack((self.gp.X, X))
            Y = np.vstack((self.gp.Y, Y))
        self.gp = self._fit_gp(X, Y)

    @property
    def n_observations(self):
        """ Returns the number of observed samples """
        if self.gp is None:
            return 0
        return self.gp.num_data

