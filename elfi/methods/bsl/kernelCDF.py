from scipy import stats as sc


def kernelCDF(x, kernel="gaussian"):
    return sc.norm.cdf(x)
