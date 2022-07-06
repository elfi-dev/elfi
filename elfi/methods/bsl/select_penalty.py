"""Select penalty for glasso and warton shrinkage."""

import logging

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

import elfi
from elfi.methods.bsl.pdf_methods import gaussian_syn_likelihood, semi_param_kernel_estimate
from elfi.methods.utils import batch_to_arr2d
from elfi.model.elfi_model import ElfiModel, Summary

logger = logging.getLogger(__name__)


@ignore_warnings(category=ConvergenceWarning)  # graphical lasso bad values
def select_penalty(model, batch_size, theta, summary_names=None, lmdas=None,
                   M=20, sigma=1.5, method="bsl", shrinkage="glasso",
                   whitening=None, seed=None, verbose=False):
    """Select the penalty value to use within an MCMC BSL algorithm.

    Selects the penalty (lambda) value that gives the closest estimated
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
    summary_names : str or optional, 
        Summaries used in synthetic likelihood estimation. Defaults to all summary statistics.
    M : int, optional
        The number of repeats at the same lambda and batch_size values
        to estimate the stdev of the log-likelihood
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
            lmdas = list((np.arange(0.2, 0.8, 0.02)))

    n_lambda = len(lmdas)
    batch_size = np.array([batch_size]).flatten()
    ns = len(batch_size)

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.generate_state(M)

    logliks = np.zeros((M, ns, n_lambda))

    if summary_names is None:
        summary_names = [node for node in model.nodes if isinstance(model[node], Summary)
                         and not node.startswith('_')]
        logger.info('Using all summary statistics in synthetic likelihood estimation.')
    if isinstance(summary_names, str):
        summary_names = [summary_names]
    obs_ss = np.column_stack([model[summary_name].observed for summary_name in summary_names])

    if isinstance(theta, dict):
        param_values = theta
    else:
        param_values = dict(zip(model.parameter_names, theta))

    for m_iteration in range(M):  # for M logliks at same penalty and batch_size
        ssx = model.generate(max(batch_size),
                             outputs=summary_names,
                             with_values=param_values,
                             seed=child_seeds[m_iteration])
        ssx_arr = batch_to_arr2d(ssx, summary_names)
        for n_iteration in range(ns):
            idx = np.random.choice(max(batch_size),
                                   batch_size[n_iteration],
                                   replace=False)
            ssx_n = ssx_arr[idx]

            for lmda_iteration in range(n_lambda):
                sl_method_fn = _resolve_sl_method(method)
                try:
                    loglik = sl_method_fn(ssx_n,
                                          obs_ss,
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
    for i in range(ns):
        std_devs = np.array([np.std(logliks[:, i, j]) for j in range(n_lambda)])
        closest_arg = np.argmin(np.abs(std_devs - sigma))
        closest_lmdas[i] = lmdas[closest_arg]
    if verbose:
        print('logliks: ', logliks)
        print('std_devs: ', std_devs)
    return closest_lmdas

def _resolve_sl_method(sl_method):
    
    sl_method = sl_method.lower()
    if sl_method == "bsl" or sl_method == "sbsl":
        sl_method_fn = gaussian_syn_likelihood
    elif sl_method == "semibsl":
        sl_method_fn = semi_param_kernel_estimate
    elif sl_method == "ubsl":
        raise ValueError("Unbiased BSL does not use shrinkage/penalty.")
    elif sl_method == "misspecbsl" or sl_method == "rbsl":
        raise ValueError("Misspecified BSL does not use shrinkage/penalty.")
    else:
        raise ValueError("no method with name ", sl_method, " found")

    return sl_method_fn

