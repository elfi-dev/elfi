import scipy
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import multivariate_normal
from scipy.stats import norm

import scipy.stats as ss
import matplotlib.pyplot as plt
import elfi
from scipy.stats import norm
# %matplotlib inline

import logging
logging.basicConfig(level=logging.INFO)


def simulator(mu_val, sigma_val, batch_size=1, random_state=None):
    mu_val = mu_val.reshape((batch_size, -1))
    n_dim = mu_val.shape[1]
    sigma_val = sigma_val.reshape((batch_size, n_dim, n_dim))

    return_mat = np.empty((batch_size, 1000, n_dim))
    for i in range(batch_size):
        return_mat[i, :, :] = multivariate_normal.rvs(mu_val[i], sigma_val[i], size=1000, random_state=random_state)

    return return_mat


def simulator_syn(mu_val, batch_size=1, random_state=None):
    mu_val = mu_val.reshape((batch_size, -1))
    n_dim = mu_val.shape[1]
    sigma_val = np.eye(n_dim)
    return_mat = multivariate_normal.rvs(np.zeros(n_dim), sigma_val, size=batch_size, random_state=random_state)+mu_val

    # return_mat = np.empty((batch_size, 1000, n_dim))
    # for i in range(batch_size):
    #     return_mat[i, :, :] = multivariate_normal.rvs(mu_val[i], sigma_val[i], size=1000, random_state=random_state)

    return return_mat

def simulator_multivariate(muss, batch_size=1, random_state=None):
    mu_ss = muss.reshape((batch_size, -1))

    Sigma1 = np.ones((mu_ss.shape[1], mu_ss.shape[1]))*0.5
    np.fill_diagonal(Sigma1, 1)

    return_mat = multivariate_normal.rvs(np.zeros(mu_ss.shape[1]), Sigma1, size=batch_size)+muss

    return return_mat


def simulator_bivariate(mu_ii, mu_jj, batch_size=1, random_state=None):
    mu_i = mu_ii.reshape((batch_size, -1))
    mu_j = mu_jj.reshape((batch_size, -1))

    Sigma1 = np.ones((2, 2))*0.5
    np.fill_diagonal(Sigma1, 1)

    return_mat = multivariate_normal.rvs(np.zeros(2), Sigma1, size=batch_size)+np.hstack((mu_i, mu_j))

    return return_mat

def simulator_univariate(mu_ii, batch_size=1, random_state=None):
    mu_i = mu_ii.reshape((-1))

    Sigma1 = 1

    return_mat = norm.rvs(0, Sigma1, size=batch_size)+mu_i

    return return_mat


def dist(val1, val2):
    dist = (np.sum((val1-val2)**2, axis=1))**0.5
    return dist

def ghat(x, data):
    nn = len(data)
    h = 1.06*np.std(data)*(nn**(-0.2))
    return np.sum(norm.pdf((x.reshape((-1, 1))-data)/h), axis = 1)/(nn*h)

def Ghat(x, data):
    nn = len(data)
    h = 1.06*np.std(data)*(nn**(-0.2))
    # x = np.array([-2.5, 1.5, -0.7, 0.3])
    return np.sum(norm.cdf((x.reshape((-1, 1))-data)/h), axis = 1)/(nn)

def mean(y):
    # return y
    return np.mean(y, axis=1)

def var(y):
    return np.var(y, axis=1)

def identity(y):
    return y

def identity_0(y):
    if len(y.shape)==1:
        return y[0]
    else:
        return y[:, 0]

def identity_1(y):
    if len(y.shape)==1:
        return y[1]
    else:
        return y[:, 1]



def gaussian_copula(thetas, margs, Lambdas):

    etas = np.zeros(thetas.shape)
    pp = len(margs[0])
    LambdasInv = inv(Lambdas)
    for ii in range(pp):
        etas[:, ii] = norm.ppf(Ghat(thetas[:, ii], margs[:, ii]))
    # temp = np.zeros(len(thetas))
    # for ii in range(len(thetas)):
    #     temp[ii] = 0.5*np.dot(np.dot(etas[ii], np.eye(pp)-LambdasInv), etas[ii].T)
    temp = 0.5*np.diag(np.dot(np.dot(etas, np.eye(pp)-LambdasInv), etas.T))

    gs = np.zeros(thetas.shape)
    for ii in range(pp):
        gs[:, ii] = np.log(ghat(thetas[:, ii], margs[:, ii]))
    gsum = np.sum(gs, axis=1)

    return np.exp(temp+gsum-0.5*np.log(det(Lambdas)))


# def simulator(mu_val, sigma_val, batch_size=1, random_state=None):
#     mu_val = mu_val.reshape((batch_size, -1))
#     n_dim = mu_val.shape[1]
#     sigma_val = sigma_val.reshape((batch_size, n_dim, n_dim))
#
#     return_mat = np.empty((batch_size, 1000, n_dim))
#     # multivariate_normal.rvs(mu_val, sigma_val).shape
#     #
#     # sigma_val.shape
#     # mu_val.shape
#
#
#     for i in range(batch_size):
#         return_mat[i, :, :] = multivariate_normal.rvs(mu_val[i], sigma_val[i], size=1000, random_state=random_state)
#
#     return return_mat

    #
    # mu_val, sigma_val = np.atleast_1d(mu_val, sigma_val)
    # if batch_size == 1:
    #     return_mat = multivariate_normal.rvs(mu_val, sigma_val, size = 1000, random_state=random_state)
    # elif batch_size >1:
    #     aa = multivariate_normal.rvs(np.zeros(len(sigma_val[0])), np.diag(np.ones(len(sigma_val[0]))), size = batch_size)
    #     return_mat = np.einsum('ijk, ik->ij', sigma_val, aa)
