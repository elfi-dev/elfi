import numpy as np
import GPy

from .regressionmodels import RegressionModelBase

class GpyModel(RegressionModelBase):

    gp_kernel_type = "exponential"
    gp_kernel_var = 1.0
    gp_kernel_scale = 1.0
    gp_noise_var = 1.0
    # optimizers: lbfgsb, simplex, scg, adadelta, rasmussen
    optimizer = "lbfgsb"
    opt_max_iters = 1e5

    def __init__(self, n_var=0, bounds=None):
        self.gp = None
        super(GpyModel, self).__init__(n_var, bounds)

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

    def _update(self):
        """ Internal function to update GP based on observations """
        old_gp = self.gp
        self.gp = GPy.models.GPRegression(X=self.Xobs,
                                          Y=self.Yobs,
                                          kernel=self._get_kernel(),
                                          noise_var=self.gp_noise_var)
        try:
            self.gp.optimize(self.optimizer, max_iters=self.opt_max_iters)
        except np.linalg.linalg.LinAlgError as e:
            logger.critical("Numerical error in GP optimization! Reverting to previous model without latest observation.")
            self.gp = old_gp


