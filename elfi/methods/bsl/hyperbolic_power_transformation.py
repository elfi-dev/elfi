import numpy as np


def sech(x):
    """Helper function for transformation KDE

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 1/(np.cosh(x))


def hyperbolic_power_transformation(s, nu, psi, lmda):
    s = np.array(s)
    return nu * np.sinh(psi * s) * np.power(sech(psi * s), lmda)/psi
