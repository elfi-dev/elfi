
"""
Implements different BSL methods that estimate the approximate posterior
"""
import numpy as np
import scipy.stats as ss
from scipy.special import loggamma, ndtr
import scipy.optimize
import math
import pandas as pd # TODO: REMOVE - use for debugging
# from copulas.univariate import GaussianKDE as GaussKDE
# from copulas.multivariate import GaussianMultivariate
# from copulae import NormalCopula
from elfi.methods.bsl.kernelCDF import kernelCDF as krnlCDF
from elfi.methods.bsl.gaussian_copula_density import gaussian_copula_density
from elfi.methods.bsl.gaussian_rank_corr import gaussian_rank_corr as grc
from sklearn.covariance import graphical_lasso  # TODO: replace with skggm?
from elfi.methods.bsl.cov_warton import cov_warton, corr_warton
from elfi.methods.bsl.gaussian_copula_density import p2P
from elfi.methods.bsl.slice_gamma_mean import slice_gamma_mean
from elfi.methods.bsl.slice_gamma_variance import slice_gamma_variance
from elfi.methods.bsl.hyperbolic_power_transformation import hyperbolic_power_transformation
from elfi.methods.bsl.eval_loglik_tkde_params import eval_loglik_tkde_params
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

    try:
        loglik = ss.multivariate_normal.logpdf(
            ssy,
            mean=sample_mean,
            cov=sample_cov) 
    except np.linalg.LinAlgError:
        loglik = -math.inf
        
    print('loglik', loglik)
    print('self.prior.logpdf(x)', self.prior.logpdf(x))
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
    except LinALgError:
        print('Matrix is not positive definite')
        loglik = -math.inf 

    # TODO: add shrinkage, etc here or refactor?

    return loglik + self.prior.logpdf(x)


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
    # TODO?: difference jacobian between matlab and paper?
    res = nu * (1 - lmda * (np.tanh(psi*s) ** 2)) * sech(psi*s) ** (lmda - 1)
    res2 = nu*(1-lmda*np.tanh(psi*s)**2)*sech(psi*s)**(lmda-1)
    return res


def semi_param_kernel_estimate(self, x, ssx, shrinkage=None, penalty=None,
                               whitening=None): #, kernel="gaussian"):
    # self.observed = self.observed.flatten()
    # dim_ss = len(self.observed)
    
    # df = pd.DataFrame(data=ssx)
    # df.to_csv('toad_ssx.csv')
    # ssx = pd.read_csv("toad_ssx.csv")
    # ssx = ssx.to_numpy()[:, 0:48]  #  TODO: delete (obviously)
    # ssy = pd.read_csv("y_obs.csv")
    # ssy = ssy.to_numpy().flatten()
    # self.observed = ssy[0:48]  # TODO: yikes
    # print(1/0)
    # ssx = pd.read_csv("elfi/methods/bsl/ma2_ssx.csv")
    # ssx = ssx.drop(labels=['Unnamed'], axis=1, inplace=False)
    # ssx = ssx.to_numpy()
    # ssx = np.delete(ssx, 0, axis=1)
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

    jacobian1 = 1  # TODO: Leave? only for TKDE
    jacobian2 = 1  # TODO: Leave? only for TKDE
    for j in range(ns):
        ssx_j = ssx[:, j].flatten()
        # ssx_j = np.array([])
        # ssx_j = np.genfromtxt('elfi/methods/bsl/random_ssxj.csv')
        # ssx_j = ssx_j.flatten()
        # ssx_j = ssx_j[~np.isnan(ssx_j)]
        if self.tkde is not None:
            # median centre? -> what to median centre
            # print('ssx_j', ssx_j)
            # print('obs', self.observed[j])
            median = np.median(ssx_j)
            ssx_j_centred = ssx_j - median
            ssx_j_pos = [x for x in ssx_j_centred if x > 0]
            ssx_j_neg = [x for x in ssx_j_centred if x <= 0]
            ssx_j_pos = np.array(ssx_j_pos)
            ssx_j_neg = np.array(ssx_j_neg)

            if self.tkde == "tkde1":  # pos skewed
                ssx_j_min = min(ssx_j)
                ssy_min = self.observed[j]  # TODO?: vectorise all of this semiBsl step?
                shift = 0
                if ssy_min < ssx_j_min:
                    shift = ssx_j_min - ssy_min + 1
                ssx_j = np.log(1 + ssx_j - ssx_j_min + shift)
                median = np.median(ssx_j)
                ssx_j_centred = ssx_j - median
                ssx_j_pos = [x for x in ssx_j_centred if x > 0]
                ssx_j_neg = [x for x in ssx_j_centred if x <= 0]

            if self.tkde == "tkde2":  # neg skewed
                ssx_j_max = max(ssx_j)
                ssy_max = self.observed[j]  # TODO?: vectorise all of this semiBsl step?
                shift = 0
                if ssy_max > ssx_j_max:
                    shift = ssy_max - ssx_j_max + 1
                ssx_j_centred = -np.log(1 - ssx_j + ssx_j_max + shift)
                median = np.median(ssx_j)
                ssx_j_centred = ssx_j - median
                ssx_j_pos = [x for x in ssx_j_centred if x > 0]
                ssx_j_neg = [x for x in ssx_j_centred if x <= 0]

            if self.tkde == "tkde3":   # symmetric and heavy kurtosis
                ssx_j_pos = np.log(1 + ssx_j_pos)
                ssx_j_neg = -np.log(1 - ssx_j_neg)

            ssx_j_pos = np.array(ssx_j_pos)  # TODO?: why need to do this
            ssx_j_neg = np.array(ssx_j_neg)

            # Find initial parameters for optimisation
            # quantile approach in # TODO?: cite Tsai
            q = 0.95
            x_q = np.quantile(ssx_j_centred, q)
            x_p = x_q / 2
            x_p_less = [x for x in ssx_j_centred if x < x_p]

            p = len(x_p_less)/len(ssx_j_centred)
            z_p = ss.norm.ppf(p)
            z_q = ss.norm.ppf(q)

            #  Using rule in tsai...
            if ((z_q/z_p) > (x_q/x_p)):  # check for low kurtosis
                psi_p = np.arccosh(z_q/(2*z_p))/np.abs(x_p)
            else:
                psi_p = np.arccosh(z_p/(z_q - z_p))/np.abs(x_p)

            # Note: High kurtosis -> lambda close 1; low kurtosis -> close 0
            lmda_p = (np.log(z_p/z_q) + np.log(np.sinh(psi_p * x_q)) -
                      np.log(np.sinh(psi_p * x_p))) / \
                     (np.log(sech(psi_p * x_p)) - np.log(sech(psi_p * x_q)))

            # print('psi_p', psi_p)
            # print('lmda_p', lmda_p)

            # TODO: duplicate code below for negative?
            q = 0.05
            x_q = np.quantile(ssx_j_centred, q)
            x_p = x_q / 2
            x_p_less = [x for x in ssx_j_centred if x < x_p]
            p = len(x_p_less)/len(ssx_j_centred)

            # TODO: CHECK THIS FIX - ABS Negatives for logs
            # print('x_q', x_q)
            # print('x_p', x_p)
            x_q = np.abs(x_q)
            x_p = np.abs(x_p)

            z_p = ss.norm.ppf(p)
            z_q = ss.norm.ppf(q)

            #  Using rule in tsai...
            if ((z_q/z_p) > (x_q/x_p)):  # check for low kurtosis
                psi_n = np.arccosh(z_q/(2*z_p))/np.abs(x_p)
            else:
                psi_n = np.arccosh(z_p/(z_q - z_p))/np.abs(x_p)
            # print(1/0)
            # Note: High kurtosis -> lambda close 1; low kurtosis -> close 0
            lmda_n = (np.log(z_p/z_q) + np.log(np.sinh(psi_n * x_q)) -
                      np.log(np.sinh(psi_n * x_p))) / \
                     (np.log(sech(psi_n * x_p)) - np.log(sech(psi_n * x_q)))

            # perform optimisation - psi, lamdas
            # TODO: log transforms for unbounded param vals

            # print('lmda_p', lmda_p)
            # print('lmda_n', lmda_n)

            psi_p_trans = np.log(psi_p)  # psi_p > 0
            lmda_p_trans = np.log((lmda_p+1)/(1-lmda_p))  # lmda <= 1

            init_params_pos = [psi_p_trans, lmda_p_trans]
            # print('init_params_pos', init_params_pos)
            res_pos = scipy.optimize.fmin(func=eval_loglik_tkde_params,
                                          x0=init_params_pos,
                                          args=(ssx_j_pos,),
                                          maxiter=200)
            # print('res_pos', res_pos)
            psi_p_mle = np.exp(res_pos[0])
            lmda_p_mle = 1
            if not np.isposinf(lmda_p_mle):
                lmda_p_mle_trans = res_pos[1]
                lmda_p_mle = (np.exp(lmda_p_mle_trans) - 1) / \
                             (1 + np.exp(lmda_p_mle_trans))
            
            # print('psi_p_mle', psi_p_mle)
            # print('lmda_p_mle', lmda_p_mle)
            # TODO: use simplex method?
            # res = minimize(eval_loglik_tkde_params, x0=[ssx_j_pos, ])
            # print('psi_n', psi_n)
            psi_n_trans = np.log(psi_n)  # psi_p > 0
            lmda_n_trans = np.log((lmda_n+1)/(1-lmda_n))  # lmda <= 1

            init_params_neg = [psi_n_trans, lmda_n_trans]
            
            res_neg = scipy.optimize.fmin(func=eval_loglik_tkde_params,
                                          x0=init_params_neg,
                                          args=(ssx_j_neg,),
                                          maxiter=200)
            # print('res_neg', res_neg)
            # print(1/0)
            psi_n_mle = np.exp(res_neg[0])
            lmda_n_mle = 1

            if not np.isposinf(lmda_n_mle):
                lmda_n_mle_trans = res_neg[1]
                lmda_n_mle = (np.exp(lmda_n_mle_trans) - 1) / \
                             (1 + np.exp(lmda_n_mle_trans))

            N = len(ssx_j_centred)
            nu_mle = ( (1/N) * (np.sum((((np.sinh(psi_p_mle*ssx_j_pos) *
            sech(psi_p_mle*ssx_j_pos) ** lmda_p_mle))/psi_p_mle) ** 2) +
            np.sum((((np.sinh(psi_n_mle*ssx_j_neg)*sech(psi_n_mle*ssx_j_neg)**lmda_n_mle))/psi_n_mle)**2) ) )**(-0.5)
            
            # nu_mle = (1/N * np.sum(((np.sinh(psi_p_mle * ssx_j_pos) *
            #                         np.power(
            #                                 sech(psi_p_mle * ssx_j_pos),
            #                                 lmda_p_mle
            #                                 )
            #                          ) /
            #                        psi_p_mle) ** 2) +
            #           np.sum(((np.sinh(psi_n_mle * ssx_j_neg) *
            #                   np.power(
            #                         sech(psi_n_mle * ssx_j_neg),
            #                         lmda_n_mle
            #                   )
            #                   ) / psi_n_mle) ** 2)) ** -0.5
            ssx_j_pos_trans = hyperbolic_power_transformation(ssx_j_pos,
                                                              nu_mle,
                                                              psi_p_mle,
                                                              lmda_p_mle)
            
            ssx_j_neg_trans = hyperbolic_power_transformation(ssx_j_neg,
                                                              nu_mle,
                                                              psi_n_mle,
                                                              lmda_n_mle)
            y = self.observed[j]
            median = np.median(ssx_j)
            # print('y', y)
            y_centred = y - median



            # jacobian = np.zeros(len(self.observed))
            if self.tkde == "tkde0":
                # print('nu_mle', nu_mle)
                # print('lmda_p_mle', lmda_p_mle)
                # print('psi_p_mle', psi_p_mle)
                if y_centred > 0:
                    jacobian1 = jacobian_hpt(y_centred,
                                            nu_mle,
                                            lmda_p_mle,
                                            psi_p_mle)
                else:
                    jacobian2 = jacobian_hpt(y_centred,
                                            nu_mle,
                                            lmda_n_mle,
                                            psi_n_mle)

            if self.tkde == "tkde1":
                pass

            if self.tkde == "tkde2":
                y = -np.log(1-y+max(ssx_j_centred)+shift-median)
                # print('yyyy', y)
                # print(1/0)  # TODO: IN PROGRESS
            if self.tkde == "tkde3":
                pass

            # jacobian[ssy_pos_idx] = jacobian_hpt(ssy_pos,
            #                                      nu_mle,
            #                                      lmda_p_mle,
            #                                      psi_p_mle)
            # jacobian[ssy_neg_idx] = jacobian_hpt(ssy_neg,
            #                                      nu_mle,
            #                                      lmda_n_mle,
            #                                      psi_n_mle)
            ssx_j = np.concatenate((ssx_j_pos_trans, ssx_j_neg_trans))
            # print('YYYY', y)
            # print('jacobian', jacobian1)
            # print('jacobian', jacobian2)

        def calc_silverman_rule_of_thumb(gauss_kde):
            ssx_j = gauss_kde.dataset
            std_sample = np.std(ssx_j)
            iqr = ss.iqr(ssx_j)
            N = gauss_kde.n
            res = 0.9 * np.minimum(std_sample, iqr/1.34) * np.power(N, -0.2)
            # print('res', res)
            return res

        kernel = ss.kde.gaussian_kde(ssx_j, bw_method=calc_silverman_rule_of_thumb)  # silverman = nrd0


        # kernel.set_bandwidth(calc_silverman_rule_of_thumb)

        y = self.observed[j]
        if self.tkde:
            median = np.median(self.observed)
            y_centred = y - median
            y_eval = 0
            if y < 0:
                y_eval = hyperbolic_power_transformation(y_centred,
                                                        nu_mle,
                                                        psi_p_mle,
                                                        lmda_p_mle)
            else:
                y_eval = hyperbolic_power_transformation(y_centred,
                                                nu_mle,
                                                psi_n_mle,
                                                lmda_n_mle)
            y = y_eval
        # TODO: below line... check jacobian good for logs?
        pdf_y[j] = kernel.logpdf(y) * np.abs(jacobian1) * np.abs(jacobian2) 
        cov_factor = kernel.covariance_factor()
        std_sample = np.std(ssx_j)
        test_factor = cov_factor * std_sample
        test_factor_std = np.sqrt(test_factor)
        # debug_kernel_factors2[j] = cov_factor
        # debug_kernel_factors3[j] = test_factor
        # bootleg_silverman_rule = 0.9 * np.minimum(std_sample, ss.iqr(ssx_j)/1.34) * np.power(len(ssx_j), -0.2)
        # print('bootleg_silverman_rule', bootleg_silverman_rule)
        silverman_bw = calc_silverman_rule_of_thumb(kernel)
        y_u[j] = np.mean(ndtr((y - ssx_j) / kernel.factor))
        # test_box = [kernel.integrate_box_1d(np.NINF, ssx_j[i]) for i in range(n)]
        # y_u[j] = np.mean([kernel.integrate_box_1d(np.NINF, ssx_j[i]) for i in range(n)])
        # y_u[j] = kernel.integrate_box_1d()
        # TODO? use this instead?

    rho_hat = grc(ssx)
    if shrinkage is not None:
        if shrinkage == "glasso":
            # TODO: add in standardise option
            rho_hat = np.atleast_2d(rho_hat)
            gl = graphical_lasso(rho_hat, alpha=penalty)  # TODO: corr?
            rho_hat = gl[0]

        if shrinkage == "warton":
            # print('rho_hat', rho_hat.shape)
            # print('ns', ns)
            # rho_hat = p2P(rho_hat, ns) # convert to array form
            rho_hat = cov_warton(rho_hat, penalty) # TODO: corr?

    # TODO: handle shrinkage not in options
    gaussian_pdf = gaussian_copula_density(rho_hat, y_u, sd, whitening) # currently logpdf
    # pdf_y[np.where(pdf_y < 1e-30)] = 1e-30  # TODO?: check now using logpdf
    pdf = gaussian_pdf + np.sum(pdf_y) # TODO? now assumes pdf_y logpdf
    # if np.isnan(pdf):
    #     raise Exception("nans")
    #     pdf = -1e+30
    # print(1/0)
    # prior_log = self.prior.logpdf(x)
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