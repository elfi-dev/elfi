"""Helper to pick the penalty for glasso and warton shrinkage"""

import numpy as np
import elfi


def select_penalty(model, batch_size, theta, summary_names, lmdas=None,
                   M=20, sigma=1.5, method="bsl", shrinkage="glasso",
                   whitening=None, standardise=False, seed=None, verbose=False,
                   *args, **kwargs):
    """Selects the penalty (lambda) value that gives the closest estimated
       loglik standard deviation closest to sigma for each specified
       batch_size.

    Parameters
    ----------
        model : elfi.ElfiModel
            The ELFI graph used by the algorithm
        batch_size : int, np.array
            Then number of simulations. If array finds closest penalty of
            each batch_size
        theta : np.array
            Parameter point where all loglikelihoods are calculated.
        M : int, optional
            The number of repeats at the same lambda and batch_size values
            to estimate the stdev of the log-likelihood
        summary_names : np.array, str
            Names of the summary nodes in the model that are to be used
            for the BSL parametric approximation.
        lmdas : np.array, optional
            The penalties values to test over
        sigma : float
            A given standard deviation value (should be between 1 and 2)
            where the lambda value with the closest estimated loglik stdev
            to sigma is returned.
        method : str, optional
            Specifies the bsl method to approximate the likelihood.
            Defaults to "bsl".
        shrinkage : str, optional
            The shrinkage method to be used with the penalty param.
        whitening : np.array of shape (m x m) - m = num of summary statistics
            The whitening matrix that can be used to estimate the sample
            covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
            transformation helps decorrelate the summary statistics allowing
            for heaving shrinkage to be applied (hence smaller batch_size).
        seed : int, optional
            Seed for the data generation from the ElfiModel
        verbose : bool, optional
            Option to display additional information on stdevs
    Returns
    -------
        The closest lambdas (for each batch_size passed in)
    """
    if lmdas is None:
        if shrinkage == "glasso":
            lmdas = list(np.exp(np.arange(-5.5, -1.5, 0.2)))
        if shrinkage == "warton":
            lmdas = list((np.arange(0.3, 0.8, 0.02)))

    n_lambda = len(lmdas)
    batch_size = np.array([batch_size]).flatten()
    ns = len(batch_size)
    original_seed = seed if seed is not None else 0

    logliks = np.zeros((M, ns, n_lambda))

    for m_iteration in range(M):
        for n_iteration in range(ns):
            for lmda_iteration in range(n_lambda):
                seed = original_seed + m_iteration*1000 + lmda_iteration
                m = model.copy()
                bsl_temp = elfi.BSL(m, summary_names=summary_names,
                                    method=method,
                                    batch_size=batch_size[n_iteration],
                                    penalty=lmdas[lmda_iteration],
                                    shrinkage=shrinkage,
                                    whitening=whitening,
                                    standardise=standardise,
                                    seed=seed
                                    )
                logliks[m_iteration, n_iteration, lmda_iteration] = \
                    bsl_temp.select_penalty_helper(theta)

    # choose the lambda with the empirical s.d. of the log SL estimates
    # closest to sigma
    closest_lmdas = np.zeros(ns)
    for i in range(ns):
        std_devs = np.array([np.std(logliks[:, i, j]) for j in range(n_lambda)])
        closest_lmda = np.min(np.abs(std_devs - sigma))
        closest_arg = np.argmin(np.abs(std_devs - sigma))
        closest_lmdas[i] = lmdas[closest_arg]
    if verbose:
        print('logliks: ', logliks)
        print('std_devs: ', std_devs)
    return closest_lmdas
