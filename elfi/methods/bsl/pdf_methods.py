
"""
Implements different BSL methods that estimate the approximate posterior
"""
import numpy as np
import scipy.stats as ss
from scipy.special import loggamma
import math
# import pandas as pd
# from copulas.univariate import GaussianKDE as GaussKDE
# from copulas.multivariate import GaussianMultivariate
# from copulae import NormalCopula
from .kernelCDF import kernelCDF as krnlCDF
from .gaussian_copula_density import gaussian_copula_density
from .gaussian_rank_corr import gaussian_rank_corr as grc
from sklearn.covariance import graphical_lasso # TODO: replace with skggm?
from .cov_warton import cov_warton, corr_warton
from .gaussian_copula_density import p2P
from .slice_gamma_mean import slice_gamma_mean
from .slice_gamma_variance import slice_gamma_variance

# from copulas.multivariate import GaussianMultivariate

def gaussian_syn_likelihood(self, x, ssx, shrinkage=None, penalty=None,
                            whitening=None):
    """[summary]

    Args:
        x ([type]): [description]
        ssx ([type]): [description]

    Returns:
        [type]: [description]
    """
    #TODO: Expand places whitening matrix used
    if hasattr(self, 'observed'):
        ssy = self.observed
    elif hasattr(self, 'observed'):
        ssy = self.observed

    dim_ss = len(ssy)
    s1, s2 = ssx.shape
    # y1, y2 = self.observed.shape
    #ssx - A matrix of the simulated summary statistics. The number
    #      of rows is the same as the number of simulations per iteration.
    if s1 == dim_ss: # obs as columns # TODO: what about s1 == s2 ?
        ssx = np.transpose(ssx)
    
    # if
    sample_mean = ssx.mean(0)
    # print('sample_mean', sample_mean)
    sample_cov = np.cov(np.transpose(ssx))
    std = np.std(ssx, axis=0)
    ns, n = ssx.shape

    if shrinkage == 'glasso':  # todo? - standardise?
        if sample_cov.size == 1:  # Stop errors when only use 1 summary stat
            sample_cov = sample_cov.reshape((1, 1))
        gl = graphical_lasso(sample_cov, alpha=penalty)
        # NOTE: able to get precision matrix here as well
        sample_cov = gl[0]

    if shrinkage == 'warton':
        if whitening is not None:
            ssy = np.matmul(whitening, self.observed) # TODO: COMPUTE at start...not everytime
            ssx_tilde = np.matmul(ssx, np.transpose(whitening))
            sample_mean = ssx_tilde.mean(0)
            sample_cov = np.cov(np.transpose(ssx_tilde)) # TODO: stop cov. comp. multiple times
        # TODO: GRC
        sample_cov = cov_warton(sample_cov, penalty)
    print('ssy', ssy)
    print('sample_mean', sample_mean)
    try:
        loglik = ss.multivariate_normal.logpdf(
            ssy,
            mean=sample_mean,
            cov=sample_cov) 
    except np.linalg.LinAlgError:
        loglik = -math.inf
    return loglik + self.prior.logpdf(x)


def gaussian_syn_likelihood_ghurye_olkin(self, x, ssx):
    """[summary]

    Args:
        x ([type]): [description]
        ssx ([type]): [description]

    Returns:
        [type]: [description]
    """
    ssy = self.observed
    n, d = ssx.shape # rows - num of sims
    mu = np.mean(ssx, 0)
    sigma = np.cov(np.transpose(ssx))
    ssy = ssy.reshape((-1, 1))
    mu = mu.reshape((-1, 1))
    sub_vec = np.subtract(ssy, mu)

    psi = np.subtract((n - 1) * sigma,  (np.matmul(ssy - mu, np.transpose(ssy - mu)) / (1 - 1/n)))

    try:
        # temp = np.linalg.cholesky(psi)
        _ , logdet_sigma = np.linalg.slogdet(sigma)
        _ , logdet_psi = np.linalg.slogdet(psi)
        A = wcon(d, n-2) - wcon(d, n-1) - 0.5*d*math.log(1 - 1/n)
        B = -0.5 * (n-d-2) * (math.log(n-1) + logdet_sigma)
        C = 0.5 * (n-d-3) * logdet_psi
        loglik = -0.5*d*math.log(2*math.pi) + A + B + C
        # TODOL: 
    except:
        print('Matrix is not positive definite')
        loglik = -math.inf 

    # TODO: add shrinkage, etc here or refactor?

    return loglik + self.prior.logpdf(x)


def semi_param_kernel_estimate(self, x, ssx, shrinkage=None, penalty=None,
                               whitening=None, iteration=None): #, kernel="gaussian"):
    # self.observed = self.observed.flatten()
    # dim_ss = len(self.observed)

    # # TODO: temp fix to get to 2 dimensions
    n, ns = ssx.shape[0:2] # rows by col
    # print('n, ns', n, ns)
    # print('dim_ss', dim_ss)
    # print('ssx', ssx)
    # print(1/0)
    # y1, y2 = self.observed.shape
    #ssx - A matrix of the simulated summary statistics. The number
    #      of rows is the same as the number of simulations per iteration.
    # if n == dim_ss: # obs as columns
    #     ssx = np.transpose(ssx)

    pdf_y = np.zeros(ns)
    y_u = np.zeros(ns)
    sd = np.zeros(ns)

    for j in range(ns):
        ssx_j = ssx[:, j].flatten()
        kernel = ss.kde.gaussian_kde(ssx_j, bw_method="silverman") # silverman = nrd0
        approx_y = kernel.pdf(self.observed[j]) # TODO: might need to massage / check observed
        pdf_y[j] = approx_y
        y_u[j] = np.mean(krnlCDF((self.observed[j] - ssx_j) / kernel.factor))

    rho_hat = grc(ssx)
    if shrinkage is not None:
        if shrinkage == "glasso":
            rho_hat = np.atleast_2d(rho_hat)
            gl = graphical_lasso(rho_hat, alpha = penalty) # TODO: corr?
            rho_hat = gl[0]

        if shrinkage == "warton":
            rho_hat = p2P(rho_hat, ns) # convert to array form
            rho_hat = cov_warton(rho_hat, penalty) # TODO: corr?

        # TODO: handle shrinkage not in options
    gaussian_pdf = gaussian_copula_density(rho_hat, y_u, sd, whitening) # currently logpdf
    pdf_y[np.where(pdf_y < 1e-30)] = 1e-30  # TODO: code smell
    pdf = gaussian_pdf + np.sum(np.log(pdf_y))
    # if np.isnan(pdf):
    #     raise Exception("nans")
    #     pdf = -1e+30
    return pdf + self.prior.logpdf(x)


def syn_likelihood_misspec(self, x, ssx, loglik=None, type_misspec=None, tau=1,
                           penalty=None, whitening=None, iteration=None):
    ssy = self.observed
    s1, s2 = ssx.shape
    dim_ss = len(self.observed)
    # print('type_misspec', type_misspec)
    # TODO: HOW TO DO LOGLIK CURR HERE??
    # sample_mean = ssx.mean(0)
    # print('sample_mean', sample_mean)
    # sample_cov = np.cov(np.transpose(ssx))
    # std = np.std(ssx, axis=0)
    # print('type_misspec', type_misspec)
    # if type_misspec == "mean":
    #     sample_mean = sample_mean + std * self.gamma

    # if type_misspec == "variance":
    #     sample_cov = sample_cov + np.diag((std * self.gamma) ** 2)  # TODO: check if np.diag needed

    # loglik = ss.multivariate_normal.logpdf(
    #         ssy,
    #         mean=sample_mean,
            # cov=sample_cov)


    if self.curr_loglik is not None:  # TODO: first iteration?
        if type_misspec == "mean":
            # gamma_temp = np.zeros(s2)
            self.gamma = slice_gamma_mean(self, ssx, self.curr_loglik, gamma=self.gamma)
        if type_misspec == "variance":
            # gamma_temp = np.repeat(tau, s2)
            self.gamma = slice_gamma_variance(self, ssx, self.curr_loglik, gamma=self.gamma)

    # ssx - A matrix of the simulated summary statistics. The number
    #      of rows is the same as the number of simulations per iteration.
    if s1 == dim_ss:  # obs as columns
        ssx = np.transpose(ssx)

    sample_mean = ssx.mean(0)
    sample_cov = np.cov(np.transpose(ssx))
    print('sample_cov', sample_cov)
    self.prev_sample_mean = sample_mean
    self.prev_sample_cov = sample_cov

    std = np.std(ssx, axis=0)
    if type_misspec == "mean":
        sample_mean = sample_mean + std * self.gamma

    if type_misspec == "variance":
        sample_cov = sample_cov + np.diag((std * self.gamma) ** 2)  # TODO: check if np.diag needed

    try:
        print('ssy', ssy)
        print('sample_mean', sample_mean)
        print('sample_cov', sample_cov)
        loglik = ss.multivariate_normal.logpdf(
            ssy,
            mean=sample_mean,
            cov=sample_cov
            ) 
    except np.linalg.LinAlgError:
        loglik = -math.inf
    
    self.curr_loglik = loglik
    self.prev_std = std

    return loglik + self.prior.logpdf(x)


def wcon(k, nu):
    """log of c(k, nu) from Ghurye & Olkin (1969)

    Args:
        k ([type]): [description]
        nu ([type]): [description]

    Returns:
        [type]: [description]
    """
    loggamma_input = [0.5*(nu - x) for x in range(k)]

    cc = -k * nu / 2 * math.log(2) - k*(k-1)/4*math.log(math.pi) - \
        np.sum(loggamma(loggamma_input))
    return cc