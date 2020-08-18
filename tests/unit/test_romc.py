import numpy as np
import pytest
from elfi.methods.parameter_inference import ROMC


# def test_gt_around_theta():
#     # gt info
#     center = np.array([1, 1], dtype=np.float)
#     limits = np.array([[-1, 1], [-1, 1]], dtype=np.float)
#
#     # inputs to method
#     theta = center
#     r = 1
#     eps = 0.5
#     lim = 5.
#     step = 0.1
#     dim = 2
#
#     def func(x):
#         if np.linalg.norm(x - theta)**2 <= r**2:
#             return 0
#         else:
#             return 1
#
#     # run method
#     bb_list = gt_around_theta(theta_0=theta, func=func, lim=lim, step=step, dim=dim, eps=eps)
#     bb = bb_list[0]
#
#     # compare with ground truth
#     assert len(bb_list) == 1
#     assert np.allclose(bb.center, theta)
#     assert np.allclose(bb.limits, limits, atol=step)
#     assert np.allclose(bb.rotation, np.eye(2))
#
#
# def test_romc_jacobian():
#     # gt info
#     center = np.array([1, 1], dtype=np.float)
#     limits = np.array([[-1, 1], [-1, 1]], dtype=np.float)
#
#     # inputs to method
#     theta = center
#     r = 1
#     eps = 0.5
#     lim = 5.
#     step = 0.1
#     dim = 2
#
#     class result:
#         x = center
#         hess_inv = np.array([[1., 0.], [0., 1.]])
#
#     def func(x):
#         if np.linalg.norm(x - theta)**2 <= r**2:
#             return 0
#         else:
#             return 1
#
#     # run method
#     bb_list = romc_jacobian(res=result, func=func, lim=lim, step=step, dim=dim, eps=eps)
#     bb = bb_list[0]
#
#     # compare with ground truth
#     assert len(bb_list) == 1
#     assert np.allclose(bb.center, theta)
#     assert np.allclose(bb.limits, limits, atol=step)
#     assert np.allclose(bb.rotation, np.eye(2))
#
#
# test_gt_around_theta()
# test_romc_jacobian()
