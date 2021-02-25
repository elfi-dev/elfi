
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
from sklearn.covariance import graphical_lasso
from .cov_warton import cov_warton
# from copulas.multivariate import GaussianMultivariate

def gaussian_syn_likelihood(self, x, ssx, shrinkage, penalty=None,
                            whitening=None, iteration=None):
    """[summary]

    Args:
        x ([type]): [description]
        ssx ([type]): [description]

    Returns:
        [type]: [description]
    """
    #TODO: Expand places whitening matrix used
    dim_ss = len(self.y_obs)
    s1, s2 = ssx.shape
    # y1, y2 = self.y_obs.shape
    ssy = self.y_obs
    #ssx - A matrix of the simulated summary statistics. The number
    #      of rows is the same as the number of simulations per iteration.
    if s1 == dim_ss: # obs as columns
        ssx = np.transpose(ssx)

    # if
    sample_mean = ssx.mean(0)
    # print('sample_mean', sample_mean)
    sample_cov = np.cov(np.transpose(ssx))
    std = np.std(ssx, axis=0)
    ns, n = ssx.shape

    if shrinkage == 'glasso': # todo? - standardise?
        gl = graphical_lasso(sample_cov, alpha = penalty)
        sample_cov = gl[0]

    if shrinkage == 'warton':
        if whitening is not None:
            ssy = np.matmul(whitening, self.y_obs) # TODO: COMPUTE at start...not everytime
            ssx_tilde = np.matmul(ssx, np.transpose(whitening))
            sample_mean = ssx_tilde.mean(0)
            sample_cov = np.cov(np.transpose(ssx_tilde)) # TODO: stop cov. comp. multiple times
        # TODO: GRC
        sample_cov = cov_warton(sample_cov, penalty)
    
    if iteration == 0: # TODO: better?
        sample_mean = ssx.mean(0)
        sample_cov = np.eye(dim_ss)
    
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
    ssy = self.y_obs
    d = ssy.shape[1]  #TODO: MAGIC
    # print('dddd', d)
    n = ssx.shape[0] # rows - num of sims
    # print('ssx', ssx.shape)
    mu = np.mean(ssx, 0)
    sigma = np.cov(np.transpose(ssx))
    # print('ssy', ssy.shape)
    # print('mu', mu.shape)
    # print('n', n)
    # print('sigma', sigma.shape)
    ssy = ssy.reshape((d, 1))
    mu = mu.reshape((d, 1))
    # print('ssy - mu', ssy - mu)
    # ssy = ssy.flatten()
    # print('1', ((n - 1) * np.matmul(sigma - (ssy - mu))).shape)
    # print('2',  np.transpose(ssy - mu).shape)

    # print(' (ssy - mu)', np.matmul((ssy - mu), np.transpose(ssy - mu)))
    sub_vec = np.subtract(ssy, mu)
    # print('sub_vec', sub_vec.shape)
    # print('np.inner(ssy - mu, ssy - mu)', np.matmul(sub_vec, np.transpose(sub_vec)))
    # print('(1 - (1/n))', (1 - (1/n)))
    psi = np.subtract((n - 1) * sigma,  (np.matmul(ssy - mu, np.transpose(ssy - mu)) / (1 - 1/n)))
    # print('psi', psi.shape)
    # print('psi', psi)

    try:
        # temp = np.linalg.cholesky(psi)
        _ , logdet_sigma = np.linalg.slogdet(sigma)
        _ , logdet_psi = np.linalg.slogdet(psi)
        # print('logdet_sigma', logdet_sigma)
        # print('logdet_psi', logdet_psi)
        A = wcon(d, n-2) - wcon(d, n-1) - 0.5*d*math.log(1 - 1/n)
        B = -0.5 * (n-d-2) * (math.log(n-1) + logdet_sigma)
        C = 0.5 * (n-d-3) * logdet_psi
        loglik = -0.5*d*math.log(2*math.pi) + A + B + C
        # print('AAAA', A)
        # print('BBBB', B)
        # print('CCCC', C)
        # TODOL: 
        # print('gothere')
    except:
        print('Matrix is not positive definite')
        loglik = -math.inf 

    return loglik + self.prior.logpdf(x)

def semi_param_kernel_estimate(self, x, ssx): #, kernel="gaussian"):
    # print("ssxshape", ssx.shape)
    print('ssx', ssx.shape) # summary stats x obs.
    print('self.y_obs', self.y_obs.shape)
    ssx = np.transpose(ssx)
    n, ns = ssx.shape # rows by col
    # if ns < 2:
    #   raise Exception("needs 2 or more sims")
    # if len(self.y_obs) != ns:
    #   raise Exception("summary statistic of observed data does not match summary"
    #               "statistics of simultated")
    pdf_y = np.zeros(ns)
    y_u = np.zeros(ns)
    sd = np.zeros(ns)

    for j in range(ns):
        kernel = ss.kde.gaussian_kde(ssx[:, j] , bw_method="silverman") # silverman = nrd0
        approx_y = kernel.pdf(self.y_obs[j]) # TODO: might need to massage / check y_obs
        # sd[j] = np.std(ssx[:, j])
        # print('approx_y', approx_y)
        # print('ffff', type(f.dataset[0]))
        # print(1/0)



        # est_y = f(f.dataset[0])
        # print('f.datasetshape', f.dataset.shape)
        # print(' f.dataset[0]',  f.dataset[0].shape)
        # print('f.dataset', f.dataset[0])
        # approx_y = np.interp(self.y_obs[:, j], f.dataset[0], est_y)
        pdf_y[j] = approx_y
        # print('pdf_y[j]', pdf_y[j])
        # print('kernel.factor', kernel.factor)
        y_u[j] = np.mean(krnlCDF((self.y_obs[j] - ssx[:, j]) / kernel.factor))
        # print('y_u[j]', y_u[j])
        # print(1/0)
    # print(1/0)
    # print('sdsd', sd)
    rho_hat = grc(ssx)
    # print('rho_hat', rho_hat)
    # print('rho_hat1', rho_hat.shape)
    # rho_hat = rho_hat.flatten()
    # rho_hat = rho_hat.reshape((-1, 2))
    # print('rho_hat2', rho_hat.shape)
    # print('y_u', y_u.shape)
    # copula = NormalCopula()
    # copula.fit(rho_hat)
    gaussian_pdf = gaussian_copula_density(rho_hat, y_u, sd) # currently logpdf
    # print('gaussian_pdf', gaussian_pdf)
    # print('np.sum(np.log(pdf_y))', np.sum(np.log(pdf_y)))

    # gaussian_pdf = np.abs(gaussian_pdf)
    # print('gaussian_pdf', gaussian_pdf)
    # print('pdf_y', pdf_y)
    pdf_y[np.where(pdf_y < 1e-30)] = 1e-30
    pdf = gaussian_pdf + np.sum(np.log(pdf_y))
    if np.isnan(pdf):
        print('postpdf', pdf)
        raise Exception("nans")
        pdf = -1e+30

    # if pdf > 0:
        # pdf = -1e+30

    # print('pdf', pdf)
    # copula.fit(rho_hat) #don't know what to do about the copula...
    
    # def _transform_to_normal(self, X):
    return pdf + self.prior.logpdf(x)

def syn_likelihood_misspec():
    pass


def wcon(k, nu):
    """log of c(k, nu) from Ghurye & Olkin (1969)

    Args:
        k ([type]): [description]
        nu ([type]): [description]

    Returns:
        [type]: [description]
    """
    print('runningwcon')
    loggamma_input = [0.5*(nu - x) for x in range(k)]
    # print('np.sum(loggamma(loggamma_input))', np.sum(loggamma(loggamma_input)))
    cc = -k * nu / 2 * math.log(2) - k*(k-1)/4*math.log(math.pi) - \
         np.sum(loggamma(loggamma_input))
    print('cc', cc)
    return cc