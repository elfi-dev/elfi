"""Select penalty for glasso and warton shrinkage."""

import logging

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood
from elfi.methods.utils import batch_to_arr2d

logger = logging.getLogger(__name__)


@ignore_warnings(category=ConvergenceWarning)  # graphical lasso bad values
def select_penalty(model, n_sim, theta, feature_names, likelihood=None,
                   lmdas=None, M=20, sigma=1.5, shrinkage="glasso",
                   whitening=None, seed=None, verbose=False):
    """Select the penalty value to use within an MCMC BSL algorithm.

    Selects the penalty (lambda) value that gives the closest estimated
    loglik standard deviation closest to sigma for each specified
    batch_size.

    Parameters
    ----------
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm
    n_sim : int or np.array
        The number of simulations. If array, selects penalty for each simulation count.
    theta : dict or np.array
        Parameter point where all loglikelihoods are calculated.
    feature_names : str or list
        Features used in synthetic likelihood estimation.
    likelihood : callable, optional
        Synthetic likelihood estimation method. Defaults to gaussian_syn_likelihood.
    lmdas : np.array, optional
        The penalties values to test over
    M : int, optional
        The number of repeats at the same lambda and n_sim values
        to estimate the stdev of the log-likelihood
    sigma : float
        A given standard deviation value (should be between 1 and 2)
        where the lambda value with the closest estimated loglik stdev
        to sigma is returned.
    shrinkage : str, optional
        The shrinkage method to be used with the penalty param.
    whitening : np.array of shape (m x m) - m = num of summary statistics
        The whitening matrix that can be used to estimate the sample
        covariance matrix in 'BSL' or 'semiBsl' methods. Whitening
        transformation helps decorrelate the summary statistics allowing
        for heaving shrinkage to be applied (hence smaller simulation count).
    seed : int, optional
        Seed for the data generation from the ElfiModel
    verbose : bool, optional
        Option to display additional information on stdevs

    Returns
    -------
        The closest lambdas and standard deviation values (for each batch_size passed in)

    """
    param_values = theta if isinstance(theta, dict) else dict(zip(model.parameter_names, theta))
    feature_names = [feature_names] if isinstance(feature_names, str) else feature_names
    ssy = np.column_stack([model[node].observed for node in feature_names])

    likelihood = likelihood or gaussian_syn_likelihood

    if lmdas is None:
        if shrinkage == "glasso":
            lmdas = list(np.exp(np.arange(-5.5, -1.5, 0.2)))
        if shrinkage == "warton":
            lmdas = list((np.arange(0.2, 0.8, 0.02)))

    n_lambda = len(lmdas)
    batch_size = np.array([n_sim]).flatten()
    ns = len(batch_size)

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.generate_state(M)

    logliks = np.zeros((M, ns, n_lambda))

    for m_iteration in range(M):  # for M logliks at same penalty and batch_size
        ssx = model.generate(max(batch_size),
                             outputs=feature_names,
                             with_values=param_values,
                             seed=child_seeds[m_iteration])
        ssx_arr = batch_to_arr2d(ssx, feature_names)
        for n_iteration in range(ns):
            idx = np.random.choice(max(batch_size),
                                   batch_size[n_iteration],
                                   replace=False)
            ssx_n = ssx_arr[idx]

            for lmda_iteration in range(n_lambda):
                try:
                    loglik = likelihood(ssx_n,
                                        ssy,
                                        shrinkage=shrinkage,
                                        penalty=lmdas[lmda_iteration],
                                        whitening=whitening)
                except FloatingPointError as err:
                    logger.warning('Floating point error: {}'.format(err))
                    loglik = np.NINF
                logliks[m_iteration, n_iteration, lmda_iteration] = loglik

    # choose the lambda with the empirical s.d. of the log SL estimates
    # closest to sigma
    closest_lmdas = np.zeros(ns)
    closest_std_devs = np.zeros(ns)
    for i in range(ns):
        std_devs = np.array([np.std(logliks[:, i, j]) for j in range(n_lambda)])
        closest_arg = np.argmin(np.abs(std_devs - sigma))
        closest_lmdas[i] = lmdas[closest_arg]
        closest_std_devs[i] = std_devs[closest_arg]
    if verbose:
        print('logliks: ', logliks)
        print('std_devs: ', std_devs)
    return closest_lmdas, closest_std_devs
