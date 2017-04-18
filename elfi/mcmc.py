import numpy as np
import logging


logger = logging.getLogger(__name__)

# TODO: parallel chains, combine ESS and Rhat?, total ratio


def eff_sample_size(chains):
    """Calculates the effective sample size for 1 or more chains.

    See:

    Stan modeling language user's guide and reference manual, v. 2.14.0.

    Parameters
    ----------
    chains : np.array of shape (N,) or (M, N)
        Samples of a parameter from an MCMC algorithm. No burn-in subtracted here!

    Returns
    -------
    ess : float
    """
    chains = np.atleast_2d(chains)
    n_chains, n_samples = chains.shape
    means = np.mean(chains, axis=1)
    variances = np.var(chains, ddof=1, axis=1)

    var_between = 0 if n_chains==1 else n_samples * np.var(means, ddof=1)
    var_within = np.mean(variances)
    var_pooled = ((n_samples - 1.) * var_within + var_between) / n_samples

    # autocorrelations for lags 1..n_samples
    n_padded = int(2**np.ceil(1 + np.log2(n_samples)))
    freqs = np.fft.rfft(chains - means[:, None], n_padded)
    autocorr = np.fft.irfft(np.abs(freqs)**2)[:, :n_samples].real
    autocorr = autocorr[:, 1:] / autocorr[:, 0:1]

    estimator_sum = 0.
    lag = 0  # +1, since this is just the index
    while lag < n_samples:
        # estimate multi-chain autocorrelation using variogram
        temp = 1. - (var_within - np.mean(autocorr[:, lag])) / var_pooled

        # only use the first non-negative autocorrelations to avoid noise
        if temp >= 0:
            estimator_sum += temp
            lag += 1
        else:
            break

    ess = n_chains * n_samples / (1. + 2. * estimator_sum)

    return ess


def gelman_rubin(chains):
    """Calculates the Gelman--Rubin convergence statistic, also known as the
    potential scale reduction factor, or \hat{R}. Uses the split version, as in Stan.

    See:

    Gelman, A. and D. B. Rubin: Inference from iterative simulation using
    multiple sequences (with discussion). Statistical Science, 7:457-511, 1992.

    Stan modeling language user's guide and reference manual, v. 2.14.0.

    Parameters
    ----------
    chains : np.array of shape (M, N)
        Samples of a parameter from an MCMC algorithm, 1 row per chain. No burn-in subtracted here!

    Returns
    -------
    psrf : float
        Should be below 1.1 to support convergence, or at least below 1.2 for all parameters.
    """
    n_chains, n_samples = chains.shape

    # split chains in the middle
    n_chains *= 2
    n_samples //= 2  # drop 1 if odd
    chains = chains[:, :2*n_samples].reshape((n_chains, n_samples))

    means = np.mean(chains, axis=1)
    variances = np.var(chains, ddof=1, axis=1)

    var_between = n_samples * np.var(means, ddof=1)
    var_within = np.mean(variances)

    var_pooled = ((n_samples - 1.) * var_within + var_between) / n_samples

    # potential scale reduction factor, should be close to 1
    psrf = np.sqrt(var_pooled / var_within)

    return psrf


def nuts(n_samples, params0, target, grad_target, n_adapt=1000, accept_prob=0.5,
         max_depth=5, random_state=None):
    """No-U-Turn Sampler, an improved version of the Hamiltonian (Markov Chain) Monte Carlo sampler.

    Based on Algorithm 6 in
    Hoffman & Gelman, depthMLR 15, 1351-1381, 2014.

    Parameters
    ----------
    n_samples : int
        The number of requested samples.
    params0 : np.array
        Initial values for sampled parameters.
    target : function
        The target's log density to sample (possibly unnormalized).
    grad_target : function
        The gradient of target.
    n_adapt : int, optional
        The number of automatic adjustments to stepsize.
    accept_prob : float, optional
        Desired average acceptance probability.
    max_depth : int, optional
        Maximum recursion depth.
    random_state : np.random.RandomState
        State of pseudo-random number generator.

    Returns
    -------
    samples : np.array
        Samples from the MCMC algorithm, including those during adaptation.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    # ********************************
    # Find reasonable initial stepsize
    # ********************************
    stepsize = 1.
    momentum0 = random_state.randn(*params0.shape)
    grad0 = grad_target(params0)

    # leapfrog
    momentum1 = momentum0 + 0.5 * stepsize * grad0
    params1 = params0 + stepsize * momentum1
    momentum1 += 0.5 * stepsize * grad_target(params1)

    joint0 = target(params0) - 0.5 * momentum0.dot(momentum0)
    joint1 = target(params1) - 0.5 * momentum1.dot(momentum1)

    plusminus = 1 if np.exp(joint1 - joint0) > 0.5 else -1
    factor = 2. if plusminus==1 else 0.5
    while factor * np.exp(plusminus * (joint1 - joint0)) > 1.:
        stepsize *= factor
        if stepsize == 0. or stepsize > 1e7:  # bounds as in STAN
            raise SystemExit("NUTS: Found invalid stepsize {}.".format(stepsize))

        # leapfrog
        momentum1 = momentum0 + 0.5 * stepsize * grad0
        params1 = params0 + stepsize * momentum1
        momentum1 += 0.5 * stepsize * grad_target(params1)

        joint1 = target(params1) - 0.5 * momentum1.dot(momentum1)

    logger.debug("{}: Set initial stepsize {}.".format(__name__, stepsize))

    # Some parameters from the NUTS paper, used for adapting the stepsize
    target_stepsize = np.log(10. * stepsize)
    log_avg_stepsize = 0.
    accept_ratio = 0.  # tends to accept_prob
    shrinkage = 0.05  # controls shrinkage accept_ratio to accept_prob
    ii_offset = 10.  # stabilizes initialization
    discount = -0.75  # reduce weight of past

    # ********
    # Sampling
    # ********
    samples = np.empty((n_samples+1,) + params0.shape)
    samples[0, :] = params0
    n_diverged = 0  # counter for proposals whose error diverged
    n_total = 0  # total number of proposals

    for ii in range(1, n_samples+1):
        momentum0 = random_state.randn(*params0.shape)
        samples_prev = samples[ii-1, :]
        log_slicevar = target(samples_prev) - 0.5 * momentum0.dot(momentum0) - random_state.exponential()
        samples[ii, :] = samples_prev
        params_left = samples_prev
        params_right = samples_prev
        momentum_left = momentum0
        momentum_right = momentum0
        depth = 0
        n_ok = 1
        all_ok = True  # criteria for no U-turn, diverging error

        while all_ok and depth <= max_depth:
            direction = 1 if random_state.rand() < 0.5 else -1
            if direction == -1:
                params_left, momentum_left, _, _, params1, n_sub, sub_ok, mh_ratio, n_steps, n_diverged1 \
                = _build_tree_nuts(params_left, momentum_left, log_slicevar, direction * stepsize, depth, samples_prev, momentum0, target, grad_target, random_state)
            else:
                _, _, params_right, momentum_right, params1, n_sub, sub_ok, mh_ratio, n_steps,n_diverged1 \
                = _build_tree_nuts(params_right, momentum_right, log_slicevar, direction * stepsize, depth, samples_prev, momentum0, target, grad_target, random_state)

            if sub_ok == 1:
                if random_state.rand() < float(n_sub) / n_ok:
                    samples[ii, :] = params1  # accept proposal
            n_ok += n_sub
            n_diverged += n_diverged1
            n_total += n_steps
            all_ok = sub_ok and ((params_right - params_left).dot(momentum_left) >= 0) \
                            and ((params_right - params_left).dot(momentum_right) >= 0)
            depth += 1
            if depth > max_depth:
                logger.debug("{}: Maximum recursion depth {} exceeded.".format(__name__, max_depth))

        if ii <= n_adapt:
            accept_ratio = (1. - 1. / (ii + ii_offset)) * accept_ratio \
                           + (accept_prob - float(mh_ratio) / n_steps) / (ii + ii_offset)
            log_stepsize = target_stepsize - np.sqrt(ii) / shrinkage * accept_ratio
            log_avg_stepsize = ii**discount * log_stepsize + (1. - ii**discount) * log_avg_stepsize
            stepsize = np.exp(log_stepsize)

        elif ii == n_adapt + 1:  # final stepsize
            stepsize = np.exp(log_avg_stepsize)
            logger.debug("{}: Set final stepsize {}.".format(__name__, stepsize))

    logger.info("{}: Total acceptance ratio: {:.3f}, Diverged proposals: {}"
                .format(__name__, float(n_samples) / n_total, n_diverged))
    return samples[1:, :]


def _build_tree_nuts(params, momentum, log_slicevar, step, depth, params0, momentum0,
                     target, grad_target, random_state):
    """Recursively build a balanced binary tree needed by NUTS.

    Based on Algorithm 6 in
    Hoffman & Gelman, JMLR 15, 1351-1381, 2014.
    """

    # Base case: one leapfrog step
    if depth == 0:
        momentum1 = momentum + 0.5 * step * grad_target(params)
        params1 = params + step * momentum1
        momentum1 = momentum1 + 0.5 * step * grad_target(params1)

        term = target(params1) - 0.5 * momentum1.dot(momentum1)
        n_ok = float(log_slicevar <= term)
        sub_ok = log_slicevar < (1000. + term)  # check for diverging error

        mh_ratio = min(1., np.exp(term - target(params0) + 0.5 * momentum0.dot(momentum0)))
        return params1, momentum1, params1, momentum1, params1, n_ok, sub_ok, mh_ratio, 1., not sub_ok

    else:
        # Recursion to build subtrees, doubling size
        params_left, momentum_left, params_right, momentum_right, params1, n_sub, sub_ok, mh_ratio, n_steps, n_diverged \
        = _build_tree_nuts(params, momentum, log_slicevar, step, depth-1, params0, momentum0, target, grad_target, random_state)
        if sub_ok:
            if step < 0:
                params_left, momentum_left, _, _, params2, n_sub2, sub_ok, mh_ratio2, n_steps2, n_diverged1 \
                = _build_tree_nuts(params_left, momentum_left, log_slicevar, step, depth-1, params0, momentum0, target, grad_target, random_state)
            else:
                _, _, params_right, momentum_right, params2, n_sub2, sub_ok, mh_ratio2, n_steps2, n_diverged1 \
                = _build_tree_nuts(params_right, momentum_right, log_slicevar, step, depth-1, params0, momentum0, target, grad_target, random_state)

            if n_sub2 > 0:
                if float(n_sub2) / (n_sub + n_sub2) > random_state.rand():
                    params1 = params2  # accept move
            mh_ratio += mh_ratio2
            n_steps += n_steps2
            sub_ok = sub_ok and ((params_right - params_left).dot(momentum_left) >= 0) \
                            and ((params_right - params_left).dot(momentum_right) >= 0)
            n_sub += n_sub2
            n_diverged += n_diverged1

        return params_left, momentum_left, params_right, momentum_right, params1, n_sub, sub_ok, mh_ratio, n_steps, n_diverged


def metropolis(n_samples, params0, target, sigma_proposals, random_state=None):
    """Basic Metropolis Markov Chain Monte Carlo sampler with Gaussian proposals.

    Parameters
    ----------
    n_samples : int
        The number of requested samples.
    params0 : np.array
        Initial values for each sampled parameter.
    target : function
        The target log density to sample (possibly unnormalized).
    sigma_proposals : np.array
        Standard deviations for Gaussian proposals of each parameter.
    random_state : np.random.RandomState
        State of pseudo-random number generator.

    Returns
    -------
    samples : np.array
    """

    if random_state is None:
        random_state = np.random.RandomState()

    samples = np.empty((n_samples+1,) + params0.shape)
    samples[0, :] = params0
    target_current = target(params0)
    n_accepted = 0

    for ii in range(1, n_samples+1):
        samples[ii, :] = samples[ii-1, :] + sigma_proposals * random_state.randn(*params0.shape)
        target_prev = target_current
        target_current = target(samples[ii, :])

        if np.exp(target_current - target_prev) < random_state.rand():  # reject proposal
            samples[ii, :] = samples[ii-1, :]
            target_current = target_prev
        else:
            n_accepted += 1

    logger.info("{}: Total acceptance ratio: {:.3f}".format(__name__, float(n_accepted) / n_samples))
    return samples[1:, :]
