import numpy as np
import math
import scipy.stats as ss


def d_laplace(x, rate=1):
    n = len(x)
    return n * math.log(rate/2) - rate * np.sum(np.abs(x))


def slice_gamma_mean(self, ssx, loglik, gamma=None, tau=1, w=1, std=None,
                     max_iter=1000):
    """[summary]

    Args:
        ssx ([type]): [description]
        loglik ([type]): [description]
        gamma ([type], optional): [description]. Defaults to None.
        tau (int, optional): parameter for the Laplace prior distribution of gamma. Defaults to 1.
        w (int, optional): Step size for stepping out in the slice sampler. Defaults to 1.
        std ([type], optional): [description]. Defaults to None.
        max_iter (int, optional): [description]. Defaults to 1000.
    """\

    def log_gamma_prior(x):  # TODO: refactor?
        return d_laplace(x, rate=1/tau)

    # if
    sample_mean = self.prev_sample_mean
    # print('sample_mean', sample_mean)
    sample_cov = self.prev_sample_cov
    std = self.prev_std

    gamma_curr = gamma
    for ii, gamma in enumerate(gamma_curr):
        target = loglik + log_gamma_prior(gamma_curr) - np.random.exponential(1)

        lower = gamma - w
        upper = gamma + w

        # stepping out procedure for lower bound
        i = 0
        while (i <= max_iter):
            gamma_lower = gamma_curr
            gamma_lower[ii] = lower
            mu_lower = sample_mean + std * gamma_lower
            loglik = ss.multivariate_normal.logpdf(
                self.observed,
                mean=mu_lower,
                cov=sample_cov
                )
            prior = log_gamma_prior(gamma_lower)
            target_lower = loglik + prior
            if target_lower < target:
                break
            lower = lower - 1
            i += 1

        # stepping out procedure for upper bound
        i = 0
        while (i <= max_iter):
            gamma_upper = gamma_curr
            gamma_upper[ii] = upper
            mu_upper = sample_mean + std * gamma_upper
            loglik = ss.multivariate_normal.logpdf(
                self.observed,
                mean=mu_upper,
                cov=sample_cov
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
            sample_mean_prop = sample_mean + std * gamma_prop
            loglik = ss.multivariate_normal.logpdf(
                self.observed,
                mean=sample_mean_prop,
                cov=sample_cov
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
            # print('i', i)
    # print('loglik', loglik)  # TODO: return(gamma_curr, loglik) ?
    return gamma_curr
