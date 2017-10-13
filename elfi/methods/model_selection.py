"""This module contains methods for model comparison and selection."""

import numpy as np


def compare_models(sample_objs, model_priors=None):
    """Find posterior probabilities for different models.

    The algorithm requires elfi.Sample objects from prerun inference methods. For example the
    output from elfi.Rejection.sample is valid. The portion of samples for each model in the top
    discrepancies are adjusted by each models acceptance ratio and prior probability.

    Parameters
    ----------
    sample_objs : list of elfi.Sample
        Resulting Sample objects from prerun inference models. The objects must have an equal
        number of samples and include a valid `discrepancies` attribute.
    model_priors : array_like, optional
        Prior probability of each model. Defaults to 1 / n_models.

    Returns
    -------
    np.array
        Posterior probabilities for the considered models.

    """
    n_models = len(sample_objs)
    n0 = sample_objs[0].n_samples
    for s in sample_objs[1:]:
        if s.n_samples != n0:
            raise ValueError("The number of samples must be the same in all Sample objects.")

    # concatenate discrepancy vectors, sort and take the smallest n0
    try:
        discrepancies = np.concatenate([s.discrepancies for s in sample_objs])
    except ValueError:
        raise ValueError("All Sample objects must include valid discrepancies.")
    inds = np.argsort(discrepancies)[:n0]

    # calculate portions for each model
    p_models = np.empty(n_models)
    for i in range(n_models):
        low_bound = i * n0
        up_bound = (i + 1) * n0
        p_models[i] = np.logical_and(inds >= low_bound, inds < up_bound).sum()

        # adjust by the acceptance ratio
        p_models[i] *= n0 / sample_objs[i].n_sim

        # adjust by the prior model probability
        if model_priors is not None:
            p_models[i] *= model_priors[i]

    p_models = p_models / p_models.sum()

    return p_models
