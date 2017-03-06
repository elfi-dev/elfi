import numpy as np
import logging


logger = logging.getLogger(__name__)


def nuts(n_samples, params_init, target, grad_target, n_adapt=1000, delta=0.5):
    """
    No-U-Turn Sampler, an improved version of the Hamiltonian (Markov Chain) Monte Carlo sampler.

    Based on Algorithm 6 in
    Hoffman & Gelman, JMLR 15, 1351-1381, 2014.

    Parameters
    ----------

    Returns
    -------

    """

    # *******************************
    # Find reasonable initial epsilon
    # *******************************
    epsilon = 1.
    r0 = np.random.randn(*params_init.shape)
    grad_theta0 = grad_target(params_init)

    # leapfrog
    r_prime = r0 + 0.5 * epsilon * grad_theta0
    theta_prime = params_init + epsilon * r_prime
    r_prime += 0.5 * epsilon * grad_target(theta_prime)

    term0 = target(params_init) - 0.5 * r0.dot(r0)
    term_prime = target(theta_prime) - 0.5 * r_prime.dot(r_prime)

    a = 1 if np.exp(term_prime - term0) > 0.5 else -1
    factor = 2. if a==1 else 0.5
    while factor * np.exp(a * (term_prime - term0)) > 1.:
        epsilon *= factor
        if epsilon == 0. or epsilon > 1e7:
            raise SystemExit("Found invalid stepsize {}.".format(epsilon))

        # leapfrog
        r_prime = r0 + 0.5 * epsilon * grad_theta0
        theta_prime = params_init + epsilon * r_prime
        r_prime += 0.5 * epsilon * grad_target(theta_prime)

        term_prime = target(theta_prime) - 0.5 * r_prime.dot(r_prime)

    logger.debug("{}: Set initial stepsize {}.".format(__name__, epsilon))
    # Some parameters from the NUTS paper
    mu = np.log(10. * epsilon)
    log_epsilon_bar = 0.
    h = 0.
    gamma = 0.05
    t0 = 10.
    kappa = 0.75

    # ********
    # Sampling
    # ********
    samples = np.empty((n_samples+1,) + params_init.shape)
    samples[0, :] = params_init

    for m in range(1, n_samples+1):
        r0 = np.random.randn(*params_init.shape)
        samples_prev = samples[m-1, :]
        # u = np.random.rand() * np.exp(target(samples_prev) - 0.5 * r0.dot(r0))
        log_u = target(samples_prev) - 0.5 * r0.dot(r0) - np.random.exponential()
        samples[m, :] = samples_prev
        thetam = samples_prev
        thetap = samples_prev
        rp = r0
        rm = r0
        j = 0
        n = 1
        s = 1

        while s == 1:
            v = 1 if np.random.rand() < 0.5 else -1
            if v == -1:
                thetam, rm, _, _, theta_prime, n_prime, s_prime, alpha, n_alpha \
                = _build_tree_nuts(thetam, rm, log_u, v, j, epsilon, samples_prev, r0, target, grad_target)
            else:
                _, _, thetap, rp, theta_prime, n_prime, s_prime, alpha, n_alpha \
                = _build_tree_nuts(thetap, rp, log_u, v, j, epsilon, samples_prev, r0, target, grad_target)

            if s_prime == 1:
                if np.random.rand() < float(n_prime) / n:
                    samples[m, :] = theta_prime
            n += n_prime
            s = s_prime and ((thetap - thetam).dot(rm) >= 0) and ((thetap - thetam).dot(rp) >= 0)
            j += 1

        if m <= n_adapt:
            h = (1. - 1. / (m + t0)) * h + (delta - float(alpha) / n_alpha) / (m + t0)
            log_epsilon = mu - np.sqrt(m) / gamma * h
            log_epsilon_bar = m**(-kappa) * log_epsilon + (1. - m**(-kappa)) * log_epsilon_bar
            epsilon = np.exp(log_epsilon)

        elif m == n_adapt + 1:  # final stepsize
            epsilon = np.exp(log_epsilon_bar)
            logger.debug("{}: Set final stepsize {}.".format(__name__, epsilon))

    return samples[1:, :]


def _build_tree_nuts(theta, r, log_u, v, j, epsilon, theta0, r0, target, grad_target, delta_max=1000.):
    """Recursively build a balanced binary tree needed by NUTS.

    Based on Algorithm 6 in
    Hoffman & Gelman, JMLR 15, 1351-1381, 2014.
    """

    # Base case: one leapfrog step
    if j == 0:
        r_prime = r + 0.5 * v * epsilon * grad_target(theta)
        theta_prime = theta + v * epsilon * r_prime
        r_prime = r_prime + 0.5 * v * epsilon * grad_target(theta_prime)

        term = target(theta_prime) - 0.5 * r_prime.dot(r_prime)
        n_prime = log_u <= term
        s_prime = log_u < (delta_max + term)

        minterm = min(1., np.exp(term - target(theta0) + 0.5 * r0.dot(r0)))
        return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, minterm, 1.

    else:
        # Recursion to build subtrees
        thetam, rm, thetap, rp, theta_prime, n_prime, s_prime, alpha_prime, n_prime_alpha \
        = _build_tree_nuts(theta, r, log_u, v, j-1, epsilon, theta0, r0, target, grad_target)
        if s_prime == 1:
            if v == -1:
                thetam, rm, _, _, theta_prime_prime, n_prime_prime, s_prime_prime, alpha_prime_prime, n_prime__prime_alpha \
                = _build_tree_nuts(thetam, rm, log_u, v, j-1, epsilon, theta0, r0, target, grad_target, delta_max)
            else:
                _, _, thetap, rp, theta_prime_prime, n_prime_prime, s_prime_prime, alpha_prime_prime, n_prime__prime_alpha \
                = _build_tree_nuts(thetap, rp, log_u, v, j-1, epsilon, theta0, r0, target, grad_target, delta_max)

            if n_prime_prime > 0:
                if float(n_prime_prime) / (n_prime + n_prime_prime) > np.random.rand():
                    theta_prime = theta_prime_prime
            alpha_prime += alpha_prime_prime
            n_prime_alpha += n_prime__prime_alpha
            s_prime = s_prime_prime and ((thetap - thetam).dot(rm) >= 0) and ((thetap - thetam).dot(rp) >= 0)
            n_prime += n_prime_prime

        return thetam, rm, thetap, rp, theta_prime, n_prime, s_prime, alpha_prime, n_prime_alpha


def metropolis(n_samples, params_init, target, sigma_proposals):
    """Basic Metropolis Markov Chain Monte Carlo sampler with Gaussian proposals.

    Parameters
    ----------
    n_samples : int
        The number of requested samples.
    params_init : np.array
        Initial values for each sampled parameter.
    target : function
        The target density to sample (possibly unnormalized).
    sigma_proposals : np.array
        Standard deviations for Gaussian proposals of each parameter.

    Returns
    -------
    samples : np.array
    """
    samples = np.empty((n_samples+1,) + params_init.shape)
    samples[0, :] = params_init
    target_current = target(params_init)

    for ii in range(1, n_samples+1):
        samples[ii, :] = samples[ii-1, :] + sigma_proposals * np.random.randn(*params_init.shape)
        target_prev = target_current
        target_current = target(samples[ii, :])

        if np.exp(target_current - target_prev) < np.random.rand():  # reject proposal
            samples[ii, :] = samples[ii-1, :]

    return samples[1:, :]
