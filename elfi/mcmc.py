import numpy as np


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
    samples = np.empty((n_samples,) + params_init.shape)
    samples[0, :] = params_init
    target_current = target(params_init)

    for ii in range(1, n_samples):
        samples[ii, :] = samples[ii-1, :] + sigma_proposals * np.random.randn(*params_init.shape)
        target_prev = target_current
        target_current = target(samples[ii, :])

        if target_current / target_prev < np.random.rand():  # reject proposal
            samples[ii, :] = samples[ii-1, :]

    return samples