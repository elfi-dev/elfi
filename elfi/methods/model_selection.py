"""This module contains methods for model comparison and selection."""

import numpy as np


def compare_models(sample_objs, model_priors=None):
    """Find posterior probabilities for different models.

    The algorithm requires elfi.Sample objects from prerun inference methods. For example the
    output from elfi.Rejection.sample is valid. The portion of samples for each model in the top
    discrepancies are adjusted by each models acceptance ratio and prior probability.

    The discrepancies (including summary statistics) must be comparable so that it is
    meaningful to sort them!

    Parameters
    ----------
    sample_objs : list of elfi.Sample
        Resulting Sample objects from prerun inference models. The objects must include
        a valid `discrepancies` attribute.
    model_priors : array_like, optional
        Prior probability of each model. Defaults to 1 / n_models.

    Returns
    -------
    np.array
        Posterior probabilities for the considered models.

    """
    n_models = len(sample_objs)
    n_min = min([s.n_samples for s in sample_objs])

    # concatenate discrepancy vectors
    try:
        discrepancies = np.concatenate([s.discrepancies for s in sample_objs])
    except ValueError:
        raise ValueError("All Sample objects must include valid discrepancies.")

    # sort and take the smallest n_min
    inds = np.argsort(discrepancies)[:n_min]

    # calculate the portions of accepted samples for each model in the top discrepancies
    p_models = np.empty(n_models)
    up_bound = 0
    for i in range(n_models):
        low_bound = up_bound
        up_bound += sample_objs[i].n_samples
        p_models[i] = np.logical_and(inds >= low_bound, inds < up_bound).sum()

        # adjust by the number of simulations run
        p_models[i] /= sample_objs[i].n_sim

        # adjust by the prior model probability
        if model_priors is not None:
            p_models[i] *= model_priors[i]

    p_models = p_models / p_models.sum()

    return p_models
