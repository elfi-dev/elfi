from scipy import stats as sc
import numpy as np


def kernelCDF(x, kernel="gaussian"):
    mean = np.mean(x)
    sd = np.std(x)
    return sc.norm.cdf(x, loc=mean, scale=sd)  # could also standardise
