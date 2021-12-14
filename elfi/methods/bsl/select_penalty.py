"""Select penalty for glasso and warton shrinkage."""

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

import elfi
from elfi.model.elfi_model import ElfiModel, NodeReference


def resolve_model(model, target, default_reference_class=NodeReference):
    """Resolve model to get model and target (discrepancy) name."""
    if isinstance(model, ElfiModel) and target is None:
        raise NotImplementedError(
            "Please specify the target node of the inference method")

    if isinstance(model, NodeReference):
        target = model
        model = target.model

    if isinstance(target, str):
        target = model[target]

    if not isinstance(target, default_reference_class):
        raise ValueError('Unknown target node class')

    return model, target.name


@ignore_warnings(category=ConvergenceWarning)  # graphical lasso bad values
def select_penalty(model, batch_size, theta, lmdas=None,
                   M=20, sigma=1.5, method="bsl", shrinkage="glasso",
                   whitening=None, standardise=False, seed=None, verbose=False,
                   discrepancy_name=None):
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
    discrepancy_name : str, optional
        Specify which node to target if model is an ElfiModel.
    Returns
    -------
        The closest lambdas (for each batch_size passed in)

    """
    model, discrepancy_name = resolve_model(model, discrepancy_name)
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

    sl_node = model[discrepancy_name]
    summary_names = []
    for parent in sl_node.parents:
        summary_names.append(parent.name)

    param_values = dict(zip(model.parameter_names, theta))

    for m_iteration in range(M):  # for M logliks at same penalty and batch_size
        ssx = model.generate(max(batch_size),
                             outputs=summary_names,
                             with_values=param_values,
                             seed=original_seed+m_iteration)
        for n_iteration in range(ns):
            keys = ssx.keys()
            idx = np.random.choice(max(batch_size),
                                   batch_size[n_iteration],
                                   replace=False)
            values = [ssx[k][idx] for k in keys]
            ssx_n = dict(zip(keys, values))

            for lmda_iteration in range(n_lambda):
                sl_node.become(elfi.SyntheticLikelihood(method,
                                                        *sl_node.parents,
                                                        shrinkage=shrinkage,
                                                        penalty=lmdas[lmda_iteration],
                                                        whitening=whitening))
                try:
                    loglik = model.generate(batch_size[n_iteration],
                                            outputs=[sl_node.name],
                                            with_values=ssx_n)[sl_node.name]
                except FloatingPointError as err:
                    print('Caught Error: ', err)
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
