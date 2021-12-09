
"""
Implements different BSL methods that estimate the approximate posterior
"""
import numpy as np
import scipy.stats as ss
from scipy.special import loggamma
# import scipy.optimize
import math
from elfi.methods.bsl.gaussian_copula_density import gaussian_copula_density
from elfi.methods.bsl.gaussian_rank_corr import gaussian_rank_corr as grc
from sklearn.covariance import graphical_lasso  # TODO?: replace sklearn
from elfi.methods.bsl.cov_warton import cov_warton
from elfi.methods.bsl.slice_gamma_mean import slice_gamma_mean
from elfi.methods.bsl.slice_gamma_variance import slice_gamma_variance
import pandas as pd  # TODO! REMOVE WHEN DONE DEBUGGING
# from elfi.methods.bsl.hyperbolic_power_transformation import \
#     hyperbolic_power_transformation
# from elfi.methods.bsl.eval_loglik_tkde_params import eval_loglik_tkde_params


def gaussian_syn_likelihood(*ssx, shrinkage=None, penalty=None,
                            whitening=None, standardise=False, observed=None,
                            **kwargs):
    """Calculates the posterior logpdf using the standard synthetic likelihood

    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x
    shrinkage : str, optional
        The shrinkage method to be used with the penalty param. With "glasso"
        this corresponds with BSLasso and with "warton" this corresponds
        with wBsl.
    penalty : float, optional
        The penalty value to used for the specified shrinkage method.
        Must be between zero and one when using shrinkage method "Warton".
    whitening :  np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller batch_size).
    standardise : bool, optional
        Used with shrinkage method "glasso".
    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """
    ssx = np.column_stack(ssx)
    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()

    if whitening is not None:
        ssy = np.matmul(whitening, ssy).flatten()
        ssx = np.matmul(ssx, np.transpose(whitening))  # decorrelated sim sums

    sample_mean = ssx.mean(0)
    sample_cov = np.atleast_2d(np.cov(ssx, rowvar=False))

    if shrinkage == 'glasso':
        if standardise:
            std = np.sqrt(np.diag(sample_cov))
            ssx = (ssx - sample_mean) / std
            sample_cov = np.atleast_2d(np.cov(ssx, rowvar=False))
        gl = graphical_lasso(sample_cov, alpha=penalty, max_iter=200)
        # NOTE: able to get precision matrix here as well
        sample_cov = gl[0]

    if shrinkage == 'warton':
        # TODO: GRC ?
        sample_cov = cov_warton(sample_cov, penalty)

    try:
        loglik = ss.multivariate_normal.logpdf(
            ssy,
            mean=sample_mean,
            cov=sample_cov,
            )
    except np.linalg.LinAlgError:
        loglik = -math.inf

    return np.array([loglik])


def gaussian_syn_likelihood_ghurye_olkin(*ssx, observed=None, **kwargs):
    """Calculates the posterior logpdf using the unbiased estimator of
    the synthetic likelihood.
    # TODO? add shrinkage / etc similar to other BSL methods
    Parameters
    ----------
    ssx : np.array
        Simulated summaries at x
    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.
    """
    ssx = np.column_stack(ssx)
    n, d = ssx.shape
    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()
    mu = np.mean(ssx, 0)
    Sigma = np.cov(np.transpose(ssx))
    ssy = ssy.reshape((-1, 1))
    mu = mu.reshape((-1, 1))

    psi = np.subtract((n - 1) * Sigma,
                      (np.matmul(ssy - mu, np.transpose(ssy - mu))
                      / (1 - 1/n)))

    try:
        _, logdet_sigma = np.linalg.slogdet(Sigma)
        _, logdet_psi = np.linalg.slogdet(psi)
        A = wcon(d, n-2) - wcon(d, n-1) - 0.5*d*math.log(1 - 1/n)
        B = -0.5 * (n-d-2) * (math.log(n-1) + logdet_sigma)
        C = 0.5 * (n-d-3) * logdet_psi
        loglik = -0.5*d*math.log(2*math.pi) + A + B + C
    except np.linalg.LinAlgError:
        print('Matrix is not positive definite')
        loglik = -math.inf

    return np.array([loglik])


def sech(x):
    """Helper function for transformation KDE

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 1/(np.cosh(x))


def jacobian_hpt(s, nu, lmda, psi):
    """Calculates jacobian used in objective function in Hyperbolic
       Power Transformation

    Args:
        s ([type]): [description]
        lmda ([type]): [description]
        psi ([type]): [description]

    Returns:
        [type]: [description]
    """
    res = nu * (1 - lmda * (np.tanh(psi*s) ** 2)) * sech(psi*s) ** (lmda - 1)
    return res


# def tkde_func(self, ssx_j, observed):
#     # median centre? -> what to median centre
#     median = np.median(ssx_j)
#     ssx_j_centred = ssx_j - median
#     ssx_j_pos = [x for x in ssx_j_centred if x > 0]
#     ssx_j_neg = [x for x in ssx_j_centred if x <= 0]
#     ssx_j_pos = np.array(ssx_j_pos)
#     ssx_j_neg = np.array(ssx_j_neg)

#     if self.tkde == "tkde1":  # pos skewed
#         ssx_j_min = min(ssx_j)
#         ssy_min = observed  # TODO?: vectorise all of this semiBsl step?
#         shift = 0
#         if ssy_min < ssx_j_min:
#             shift = ssx_j_min - ssy_min + 1
#         ssx_j = np.log(1 + ssx_j - ssx_j_min + shift)
#         median = np.median(ssx_j)
#         ssx_j_centred = ssx_j - median
#         ssx_j_pos = [x for x in ssx_j_centred if x > 0]
#         ssx_j_neg = [x for x in ssx_j_centred if x <= 0]

#     if self.tkde == "tkde2":  # neg skewed
#         ssx_j_max = max(ssx_j)
#         ssy_max = observed  # TODO?: vectorise all of this semiBsl step?
#         shift = 0
#         if ssy_max > ssx_j_max:
#             shift = ssy_max - ssx_j_max + 1
#         ssx_j_centred = -np.log(1 - ssx_j + ssx_j_max + shift)
#         median = np.median(ssx_j)
#         ssx_j_centred = ssx_j - median
#         ssx_j_pos = [x for x in ssx_j_centred if x > 0]
#         ssx_j_neg = [x for x in ssx_j_centred if x <= 0]

#     if self.tkde == "tkde3":   # symmetric and heavy kurtosis
#         ssx_j_pos = np.log(1 + ssx_j_pos)
#         ssx_j_neg = -np.log(1 - ssx_j_neg)

#     ssx_j_pos = np.array(ssx_j_pos)  # TODO?: why need to do this
#     ssx_j_neg = np.array(ssx_j_neg)

#     # Find initial parameters for optimisation
#     # quantile approach in # TODO?: cite Tsai
#     q = 0.95
#     x_q = np.quantile(ssx_j_centred, q)
#     x_p = x_q / 2
#     x_p_less = [x for x in ssx_j_centred if x < x_p]

#     p = len(x_p_less)/len(ssx_j_centred)
#     z_p = ss.norm.ppf(p)
#     z_q = ss.norm.ppf(q)

#     #  Using rule in tsai...
#     if ((z_q/z_p) > (x_q/x_p)):  # check for low kurtosis
#         psi_p = np.arccosh(z_q/(2*z_p))/np.abs(x_p)
#     else:
#         psi_p = np.arccosh(z_p/(z_q - z_p))/np.abs(x_p)

#     # Note: High kurtosis -> lambda close 1; low kurtosis -> close 0
#     lmda_p = (np.log(z_p/z_q) + np.log(np.sinh(psi_p * x_q)) -
#                 np.log(np.sinh(psi_p * x_p))) / \
#                 (np.log(sech(psi_p * x_p)) - np.log(sech(psi_p * x_q)))

#     # print('psi_p', psi_p)
#     # print('lmda_p', lmda_p)

#     # TODO: duplicate code below for negative?
#     q = 0.05
#     x_q = np.quantile(ssx_j_centred, q)
#     x_p = x_q / 2
#     x_p_less = [x for x in ssx_j_centred if x < x_p]
#     p = len(x_p_less)/len(ssx_j_centred)

#     # TODO: CHECK THIS FIX - ABS Negatives for logs
#     # print('x_q', x_q)
#     # print('x_p', x_p)
#     x_q = np.abs(x_q)
#     x_p = np.abs(x_p)

#     z_p = ss.norm.ppf(p)
#     z_q = ss.norm.ppf(q)

#     #  Using rule in tsai...
#     if ((z_q/z_p) > (x_q/x_p)):  # check for low kurtosis
#         psi_n = np.arccosh(z_q/(2*z_p))/np.abs(x_p)
#     else:
#         psi_n = np.arccosh(z_p/(z_q - z_p))/np.abs(x_p)
#     # print(1/0)
#     # Note: High kurtosis -> lambda close 1; low kurtosis -> close 0
#     lmda_n = (np.log(z_p/z_q) + np.log(np.sinh(psi_n * x_q)) -
#                 np.log(np.sinh(psi_n * x_p))) / \
#                 (np.log(sech(psi_n * x_p)) - np.log(sech(psi_n * x_q)))

#     # perform optimisation - psi, lamdas
#     # TODO: log transforms for unbounded param vals

#     # print('lmda_p', lmda_p)
#     # print('lmda_n', lmda_n)

#     psi_p_trans = np.log(psi_p)  # psi_p > 0
#     lmda_p_trans = np.log((lmda_p+1)/(1-lmda_p))  # lmda <= 1

#     init_params_pos = [psi_p_trans, lmda_p_trans]
#     # print('init_params_pos', init_params_pos)
#     res_pos = scipy.optimize.fmin(func=eval_loglik_tkde_params,
#                                     x0=init_params_pos,
#                                     args=(ssx_j_pos,),
#                                     maxiter=200,
#                                     disp=False)
#     # print('res_pos', res_pos)
#     psi_p_mle = np.exp(res_pos[0])
#     lmda_p_mle = 1
#     if not np.isposinf(lmda_p_mle):
#         lmda_p_mle_trans = res_pos[1]
#         lmda_p_mle = (np.exp(lmda_p_mle_trans) - 1) / \
#                         (1 + np.exp(lmda_p_mle_trans))

#     # print('psi_p_mle', psi_p_mle)
#     # print('lmda_p_mle', lmda_p_mle)
#     # TODO: use simplex method?
#     # res = minimize(eval_loglik_tkde_params, x0=[ssx_j_pos, ])
#     # print('psi_n', psi_n)
#     psi_n_trans = np.log(psi_n)  # psi_p > 0
#     lmda_n_trans = np.log((lmda_n+1)/(1-lmda_n))  # lmda <= 1

#     init_params_neg = [psi_n_trans, lmda_n_trans]

#     res_neg = scipy.optimize.fmin(func=eval_loglik_tkde_params,
#                                     x0=init_params_neg,
#                                     args=(ssx_j_neg,),
#                                     maxiter=200,
#                                     disp=False)
#     # print('res_neg', res_neg)
#     # print(1/0)
#     psi_n_mle = np.exp(res_neg[0])
#     lmda_n_mle = 1

#     if not np.isposinf(lmda_n_mle):
#         lmda_n_mle_trans = res_neg[1]
#         lmda_n_mle = (np.exp(lmda_n_mle_trans) - 1) / \
#                         (1 + np.exp(lmda_n_mle_trans))

#     N = len(ssx_j_centred)
#     nu_mle = ( (1/N) * (np.sum((((np.sinh(psi_p_mle*ssx_j_pos) *
#     sech(psi_p_mle*ssx_j_pos) ** lmda_p_mle))/psi_p_mle) ** 2) +
#     np.sum((((np.sinh(psi_n_mle*ssx_j_neg)*sech(psi_n_mle*ssx_j_neg)**
#     lmda_n_mle))/psi_n_mle)**2) ) )**(-0.5)

#     # nu_mle = (1/N * np.sum(((np.sinh(psi_p_mle * ssx_j_pos) *
#     #                         np.power(
#     #                                 sech(psi_p_mle * ssx_j_pos),
#     #                                 lmda_p_mle
#     #                                 )
#     #                          ) /
#     #                        psi_p_mle) ** 2) +
#     #           np.sum(((np.sinh(psi_n_mle * ssx_j_neg) *
#     #                   np.power(
#     #                         sech(psi_n_mle * ssx_j_neg),
#     #                         lmda_n_mle
#     #                   )
#     #                   ) / psi_n_mle) ** 2)) ** -0.5
#     ssx_j_pos_trans = hyperbolic_power_transformation(ssx_j_pos,
#                                                         nu_mle,
#                                                         psi_p_mle,
#                                                         lmda_p_mle)

#     ssx_j_neg_trans = hyperbolic_power_transformation(ssx_j_neg,
#                                                         nu_mle,
#                                                         psi_n_mle,
#                                                         lmda_n_mle)
#     y = observed
#     median = np.median(ssx_j)
#     # print('y', y)
#     y_centred = y - median

#     if y_centred > 0:
#         jacobian = jacobian_hpt(y_centred,
#                                  nu_mle,
#                                  lmda_p_mle,
#                                  psi_p_mle)
#     else:
#         jacobian = jacobian_hpt(y_centred,
#                                  nu_mle,
#                                  lmda_n_mle,
#                                  psi_n_mle)

#     if self.tkde == "tkde1":
#         pass

#     if self.tkde == "tkde2":
#         y = -np.log(1-y+max(ssx_j_centred)+shift-median)
#     if self.tkde == "tkde3":
#         pass

#     ssx_j = np.concatenate((ssx_j_pos_trans, ssx_j_neg_trans))

#     median = np.median(self.observed)
#     y_centred = y - median
#     y_eval = 0
#     if y < 0:
#         y_eval = hyperbolic_power_transformation(y_centred,
#                                                 nu_mle,
#                                                 psi_p_mle,
#                                                 lmda_p_mle)
#     else:
#         y_eval = hyperbolic_power_transformation(y_centred,
#                                         nu_mle,
#                                         psi_n_mle,
#                                         lmda_n_mle)

#     return jacobian, ssx_j, y_eval


# def tkde_y_trans():
#     pass


def semi_param_kernel_estimate(*ssx, shrinkage=None, penalty=None,
                               whitening=None, standardise=False,
                               observed=None, tkde=False):
    """Calculates the posterior logpdf using the semi-parametric log likelihood
    of An, Z. (2020).

    References
    ----------
    An, Z., Nott, D. J., and Drovandi, C. C. (2020).
    Robust Bayesian Synthetic Likelihood via a semi-parametric approach.
    Statistics and Computing, 30:543557.

    Parameters
    ----------
    ssx :  Simulated summaries at x
    shrinkage : str, optional
        The shrinkage method to be used with the penalty param. With "glasso"
        this corresponds with BSLasso and with "warton" this corresponds
        with wBsl.
    penalty : float, optional
        The penalty value to used for the specified shrinkage method.
        Must be between zero and one when using shrinkage method "Warton".
    whitening :  np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller batch_size).
    standardise : bool, optional
        Used with shrinkage method "glasso".

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """

    ssx = np.column_stack(ssx)  # TODO: DEBUGGING
    # ssx = pd.read_csv("ssx_ma2_semibsl.csv")
    # ssx = np.array(ssx)[:, 1:]
    # ssy = pd.read_csv("ssy_ma2_semibsl.csv")
    # ssy = np.array(ssy)[:, 1:].flatten()
    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()
    dim_ss = len(ssy)

    n, ns = ssx.shape

    logpdf_y = np.zeros(ns)
    y_u = np.zeros(ns)
    sim_eta = np.zeros((n, ns))  # only used for wsemibsl
    eta_cov = None
    jacobian = 1  # used for TKDE method
    for j in range(ns):
        ssx_j = ssx[:, j].flatten()
        y = ssy[j]
        # if tkde:  # TODO? uncomment for tkde
        #     jacobian, ssx_j, y = tkde_func(ssx_j, y)

        # NOTE: bw_method - "silverman" is being used here is slightly
        #       different than "nrd0" - silverman's rule of thumb in R.
        # TODO! DECIDE BW_METHOD / KDE USED
        kernel = ss.kde.gaussian_kde(ssx_j, bw_method="silverman")
        logpdf_y[j] = kernel.logpdf(y) * np.abs(jacobian)

        y_u[j] = kernel.integrate_box_1d(np.NINF, y)

        if whitening is not None:
            # TODO!: VERY INEFFICIENT...could just use ranks?
            sim_eta[:, j] = [ss.norm.ppf(kernel.integrate_box_1d(np.NINF,
                                                                 ssx_i))
                             for ssx_i in ssx_j]

    # Below is exit point for helper function for estimate_whitening_matrix
    if not hasattr(whitening, 'shape') and whitening == "whitening":
        return sim_eta

    rho_hat = grc(ssx)

    if whitening is not None:
        # whitening_eta = np.matmul(whitening, np.transpose(sim_eta))
        eta_cov = np.cov(np.transpose(sim_eta))
        rho_hat = grc(ssx)
        rho_hat = np.matmul(rho_hat, np.transpose(whitening))

    if shrinkage == "glasso":
        # sample_mean = np.mean(ssx, axis=0)
        sample_cov = np.cov(ssx, rowvar=False)
        std = np.sqrt(np.diag(sample_cov))
        # convert from correlation matrix -> covariance
        sample_cov = np.outer(std, std) * rho_hat
        sample_cov = np.atleast_2d(sample_cov)
        gl = graphical_lasso(sample_cov, alpha=penalty)
        sample_cov = gl[0]
        rho_hat = np.corrcoef(sample_cov)

    gaussian_logpdf = gaussian_copula_density(rho_hat, y_u,
                                              penalty, whitening,
                                              eta_cov)

    pdf = gaussian_logpdf + np.sum(logpdf_y)

    if whitening is not None:
        Z_y = ss.norm.ppf(y_u)
        pdf -= np.sum(ss.norm.logpdf(Z_y, 0, 1))

    pdf = np.nan_to_num(pdf, nan=np.NINF)

    return np.array([pdf])


def syn_likelihood_misspec(self, *ssx, type_misspec=None, tau=1,
                           penalty=None, whitening=None, observed=None,
                           gamma=None, curr_loglik=None, prev_std=None,
                           **kwargs):
    """Calculates the posterior logpdf using the standard synthetic likelihood

    Parameters
    ----------
    ssx :  Simulated summaries at x
    shrinkage : str, optional
        The shrinkage method to be used with the penalty param. With "glasso"
        this corresponds with BSLasso and with "warton" this corresponds
        with wBsl.
    type_misspec : str
        String name of type of misspecified BSL. Can be either "mean" or
        "variance".
    tau : float, optional
        Scale (or inverse rate) parameter for the Laplace prior distribution of
        gamma. Defaults to 1.
    penalty : float, optional
        The penalty value to used for the specified shrinkage method.
        Must be between zero and one when using shrinkage method "Warton".
    whitening :  np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller batch_size).

    Returns
    -------
    Estimate of the logpdf for the approximate posterior at x.

    """

    ssx = np.column_stack(ssx)
    # Ensure observed are 2d
    ssy = np.concatenate([np.atleast_2d(o) for o in observed], axis=1).flatten()
    s1, s2 = ssx.shape
    dim_ss = len(ssy)

    batch_idx = kwargs['meta']['batch_index']
    prev_loglik = self.state['logliks'][batch_idx]
    prev_std = self.state['stdevs'][batch_idx]
    prev_sample_mean = self.state['sample_means'][batch_idx]
    prev_sample_cov = self.state['sample_covs'][batch_idx]
    # first iter -> does not use mean/var - adjustment
    if prev_loglik is not None:
        gamma = self.state['gammas'][batch_idx-1]
        if gamma is None:
            gamma = np.repeat(tau, dim_ss)
        if type_misspec == "mean":
            gamma = slice_gamma_mean(ssx,
                                     ssy=ssy,
                                     loglik=prev_loglik,
                                     gamma=gamma,
                                     std=prev_std,
                                     sample_mean=prev_sample_mean,
                                     sample_cov=prev_sample_cov
                                     )
        if type_misspec == "variance":
            gamma = slice_gamma_variance(ssx,
                                         ssy=ssy,
                                         loglik=prev_loglik,
                                         gamma=gamma,
                                         std=prev_std,
                                         sample_mean=prev_sample_mean,
                                         sample_cov=prev_sample_cov
                                         )

        self.update_gamma(gamma)
    if s1 == dim_ss:  # obs as columns
        ssx = np.transpose(ssx)

    sample_mean = ssx.mean(0)
    sample_cov = np.cov(ssx, rowvar=False)

    std = np.std(ssx, axis=0)
    if gamma is None:
        gamma = np.repeat(tau, dim_ss)
    if type_misspec == "mean":
        sample_mean = sample_mean + std * gamma

    if type_misspec == "variance":
        sample_cov = sample_cov + np.diag((std * gamma) ** 2)

    try:
        loglik = ss.multivariate_normal.logpdf(
            ssy,
            mean=sample_mean,
            cov=sample_cov,
            )
    except np.linalg.LinAlgError:
        loglik = -math.inf

    return loglik


def wcon(k, nu):
    """log of c(k, nu) from Ghurye & Olkin (1969)

    Args:
    k : int
    nu : int
    Returns:
    cc: float

    """
    loggamma_input = [0.5*(nu - x) for x in range(k)]

    cc = -k * nu / 2 * math.log(2) - k*(k-1)/4*math.log(math.pi) - \
        np.sum(loggamma(loggamma_input))
    return cc
