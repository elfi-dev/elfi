import scipy
import numpy as np

from elfi.model.utils import distance_as_discrepancy


from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import multivariate_normal
from scipy.stats import norm
import statsmodels.api as sm
import scipy.stats as ss
import matplotlib.pyplot as plt
import elfi
from scipy.stats import norm
# %matplotlib inline
from elfi import adjust_posterior
from elfi.methods.parameter_inference import ParameterInference

import logging
logging.basicConfig(level=logging.INFO)
from utility import *

from Inference_COPULA_ABC_inherit_rejection import Copula_ABC
import elfi


def dimension_wise_dis(X, y):
    return abs(X-y)

def run_copulaABC():
    np.random.seed(20180509)
    PP = 4 # dimensions

    yobs = np.array([[1, 2, 3, 4]])

    m = elfi.new_model()
    n_sample = 500
    quantiles = 0.01

    elfi.Prior(ss.multivariate_normal, np.zeros(PP), np.eye(PP), model = m, name = 'muss')
    elfi.Simulator(simulator_multivariate, m['muss'], observed=yobs, name = 'Gauss')

    elfi.Summary(identity, m['Gauss'], name = 'identity')
    elfi.Distance('dimension_wise', m['identity'], name = 'd')

    # rej = elfi.Rejection(m['d'], output_names=['identity'], batch_size=10000, seed=20180509).sample(n_sample, quantile = quantiles)
    rej = Copula_ABC(m['d'], output_names=['identity'], batch_size=10000, seed=20180509).sample(n_sample, quantile = quantiles)

    a = 1

if __name__ == '__main__':
    run_copulaABC()