import numpy as np
import random

from abcpy.bo.utils import approx_second_partial_derivative
from abcpy.bo.utils import sum_of_rbf_kernels

class Test_sum_of_rbf_kernels():

    def test_no_points(self):
        """ Test that in case no points are give, zero is returned """
        point = np.atleast_1d(random.uniform(0,1))
        kern_centers = np.atleast_2d()
        kern_ampl = random.uniform(0.0, 1.0)
        kern_scale = random.uniform(0.0, 1.0)
        ret = sum_of_rbf_kernels(point, kern_centers, kern_ampl, kern_scale)
        assert ret == 0.0

    def test_one_point(self):
        """ Test in case one point is given """
        point = np.atleast_1d(random.uniform(0,1))
        kern_centers = np.atleast_2d(point + 1.0)
        kern_ampl = 1.0
        kern_scale = 1.0
        ret = sum_of_rbf_kernels(point, kern_centers, kern_ampl, kern_scale)
        assert abs(ret - np.exp(-1.0)) < 1e-5

    def test_two_points(self):
        """ Test in case two points are given """
        point = np.atleast_1d(random.uniform(0,1))
        kern_centers = np.atleast_2d([[point + 1.0], [point - 1.0]])
        kern_ampl = 1.0
        kern_scale = 1.0
        ret = sum_of_rbf_kernels(point, kern_centers, kern_ampl, kern_scale)
        assert abs(ret - 2*np.exp(-1.0)) < 1e-5


class Test_approx_second_partial_derivative():

    def test_x2_mid(self):
        """ Test that normal second derivative approximation works in one dimension """
        fun = lambda x: x.dot(x)
        x0 = np.atleast_1d([random.uniform(-1,1)])
        dim = 0
        h = 0.001
        bounds = ((-1.1, 1.1),)
        ret = approx_second_partial_derivative(fun, x0, dim, h, bounds)
        assert abs(ret - 2) < 1e-5

    def test_x2_edge(self):
        """ Test that the symmetric approximation at the boundary works in one dimension """
        fun = lambda x: x.dot(x)
        x0 = np.atleast_1d([0.0])
        dim = 0
        h = 0.001
        if random.random() > 0.5:
            bounds = ((-1, 0),)
        else:
            bounds = ((0, 1),)
        ret = approx_second_partial_derivative(fun, x0, dim, h, bounds)
        assert abs(ret - 2) < 1e-3

    def test_2_dim(self):
        """ Test that normal second derivative approximation works in two dimensions """
        fun = lambda x: x.dot(x)
        x0 = np.atleast_1d([random.uniform(-1,1), random.uniform(-1,1)])
        dim = random.randint(0,1)
        h = 0.001
        bounds = ((-1.1, 1.1),(-1.1, 1.1))
        ret = approx_second_partial_derivative(fun, x0, dim, h, bounds)
        assert abs(ret - 2) < 1e-5


