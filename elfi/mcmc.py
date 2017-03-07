import numpy as np
import logging


logger = logging.getLogger(__name__)

# TODO: seed, parallel chains, max depth, Rhat, divergence


def nuts(n_samples, params0, target, grad_target, n_adapt=1000, accept_prob=0.5):
    """
    No-U-Turn Sampler, an improved version of the Hamiltonian (Markov Chain) Monte Carlo sampler.

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
    Returns
    -------
    samples : np.array
        Samples from the MCMC algorithm, including those during adaptation.
    """

    # ********************************
    # Find reasonable initial stepsize
    # ********************************
    stepsize = 1.
    momentum0 = np.random.randn(*params0.shape)
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

    for ii in range(1, n_samples+1):
        momentum0 = np.random.randn(*params0.shape)
        samples_prev = samples[ii-1, :]
        log_slicevar = target(samples_prev) - 0.5 * momentum0.dot(momentum0) - np.random.exponential()
        samples[ii, :] = samples_prev
        params_left = samples_prev
        params_right = samples_prev
        momentum_left = momentum0
        momentum_right = momentum0
        depth = 0
        n_ok = 1
        all_ok = True  # criteria for no U-turn, diverging error

        while all_ok:
            direction = 1 if np.random.rand() < 0.5 else -1
            if direction == -1:
                params_left, momentum_left, _, _, params1, n_sub, sub_ok, mh_ratio, n_steps \
                = _build_tree_nuts(params_left, momentum_left, log_slicevar, direction * stepsize, depth, samples_prev, momentum0, target, grad_target)
            else:
                _, _, params_right, momentum_right, params1, n_sub, sub_ok, mh_ratio, n_steps \
                = _build_tree_nuts(params_right, momentum_right, log_slicevar, direction * stepsize, depth, samples_prev, momentum0, target, grad_target)

            if sub_ok == 1:
                if np.random.rand() < float(n_sub) / n_ok:
                    samples[ii, :] = params1  # accept proposal
            n_ok += n_sub
            all_ok = sub_ok and ((params_right - params_left).dot(momentum_left) >= 0) \
                            and ((params_right - params_left).dot(momentum_right) >= 0)
            depth += 1

        if ii <= n_adapt:
            accept_ratio = (1. - 1. / (ii + ii_offset)) * accept_ratio \
                           + (accept_prob - float(mh_ratio) / n_steps) / (ii + ii_offset)
            log_stepsize = target_stepsize - np.sqrt(ii) / shrinkage * accept_ratio
            log_avg_stepsize = ii**discount * log_stepsize + (1. - ii**discount) * log_avg_stepsize
            stepsize = np.exp(log_stepsize)

        elif ii == n_adapt + 1:  # final stepsize
            stepsize = np.exp(log_avg_stepsize)
            logger.debug("{}: Set final stepsize {}.".format(__name__, stepsize))

    return samples[1:, :]


def _build_tree_nuts(params, momentum, log_slicevar, step, depth, params0, momentum0,
                     target, grad_target):
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
        sub_ok = log_slicevar < (1000. + term)  # diverging error

        mh_ratio = min(1., np.exp(term - target(params0) + 0.5 * momentum0.dot(momentum0)))
        return params1, momentum1, params1, momentum1, params1, n_ok, sub_ok, mh_ratio, 1.

    else:
        # Recursion to build subtrees, doubling size
        params_left, momentum_left, params_right, momentum_right, params1, n_sub, sub_ok, mh_ratio, n_steps \
        = _build_tree_nuts(params, momentum, log_slicevar, step, depth-1, params0, momentum0, target, grad_target)
        if sub_ok:
            if step < 0:
                params_left, momentum_left, _, _, params2, n_sub2, sub_ok, mh_ratio2, n_steps2 \
                = _build_tree_nuts(params_left, momentum_left, log_slicevar, step, depth-1, params0, momentum0, target, grad_target)
            else:
                _, _, params_right, momentum_right, params2, n_sub2, sub_ok, mh_ratio2, n_steps2 \
                = _build_tree_nuts(params_right, momentum_right, log_slicevar, step, depth-1, params0, momentum0, target, grad_target)

            if n_sub2 > 0:
                if float(n_sub2) / (n_sub + n_sub2) > np.random.rand():
                    params1 = params2  # accept move
            mh_ratio += mh_ratio2
            n_steps += n_steps2
            sub_ok = sub_ok and ((params_right - params_left).dot(momentum_left) >= 0) \
                            and ((params_right - params_left).dot(momentum_right) >= 0)
            n_sub += n_sub2

        return params_left, momentum_left, params_right, momentum_right, params1, n_sub, sub_ok, mh_ratio, n_steps


def metropolis(n_samples, params0, target, sigma_proposals):
    """Basic Metropolis Markov Chain Monte Carlo sampler with Gaussian proposals.

    Parameters
    ----------
    n_samples : int
        The number of requested samples.
    params0 : np.array
        Initial values for each sampled parameter.
    target : function
        The target density to sample (possibly unnormalized).
    sigma_proposals : np.array
        Standard deviations for Gaussian proposals of each parameter.

    Returns
    -------
    samples : np.array
    """
    samples = np.empty((n_samples+1,) + params0.shape)
    samples[0, :] = params0
    target_current = target(params0)

    for ii in range(1, n_samples+1):
        samples[ii, :] = samples[ii-1, :] + sigma_proposals * np.random.randn(*params0.shape)
        target_prev = target_current
        target_current = target(samples[ii, :])

        if np.exp(target_current - target_prev) < np.random.rand():  # reject proposal
            samples[ii, :] = samples[ii-1, :]

    return samples[1:, :]
