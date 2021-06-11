import numpy as np
import math
import scipy.stats as ss


def slice_gamma_variance(self, ssx, loglik, gamma=None, tau=1, w=1, std=None,
                         max_iter=1000):
    
    def log_gamma_prior(x):
        return np.sum(ss.expon.logpdf(x, scale=tau))  # tau - inv rate param, scale inv of rate.
        
    sample_mean = self.prev_sample_mean
    sample_cov = self.prev_sample_cov
    std = self.prev_std
    
    gamma_curr = gamma
    for ii, gamma in enumerate(gamma_curr):
        target = loglik + log_gamma_prior(gamma_curr) - np.random.exponential(1)

        lower = 0
        upper = gamma + w

        # stepping out procedure for upper bound
        i = 0
        while (i <= max_iter):
            gamma_upper = gamma_curr
            gamma_upper[ii] = upper
            sample_cov_upper = sample_cov + np.diag((std * gamma_upper) ** 2)
            loglik = ss.multivariate_normal.logpdf(
                self.observed,
                mean=sample_mean,
                cov=sample_cov_upper
                )
            prior = log_gamma_prior(gamma_upper)
            target_upper = loglik + prior
            if target_upper < target:
                break
            upper = upper + 1
            i += 1

        i = 0
        while (i < max_iter):
            prop = np.random.uniform(lower, upper)
            gamma_prop = gamma_curr
            gamma_prop[ii] = prop
            sample_cov_upper = sample_cov + np.diag((std * gamma_prop) ** 2)
            loglik = ss.multivariate_normal.logpdf(
                self.observed,
                mean=sample_mean,
                cov=sample_cov_upper
                )
            prior = log_gamma_prior(gamma_prop)
            target_prop = loglik + prior
            # print('target_prop', target_prop)
            # print('target', target)
            if target_prop > target:
                gamma_curr = gamma_prop
                break
            if prop < gamma:
                lower = prop
            else:
                upper = prop
            i += 1

    return gamma_curr
