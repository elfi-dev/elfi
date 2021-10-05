import numpy as np


def sech(x):
    """Helper function for transformation KDE as no numpy sech function.
    """
    return 1/(np.cosh(x))


def hyperbolic_power_transformation(s, nu, psi, lmda):
    """The hyperbolic power transform of Tsai. (2017) # TODO
    """
    s = np.array(s)
    return nu * np.sinh(psi * s) * np.power(sech(psi * s), lmda)/psi
