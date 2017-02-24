import logging
from functools import partial

import randomstate
import numpy as np

from elfi.model.elfi_model import NodeReference, ElfiModel, Discrepancy
from elfi.native_client import Client

logger = logging.getLogger(__name__)


"""Implementations of simulator based inference algorithms.
"""


class InferenceMethod(object):
    """
    """

    def __init__(self, model, batch_size=1000, seed=None):
        """

        Parameters
        ----------
        model : ElfiModel or NodeReference
        batch_size : int
        seed : int

        """
        model = model.model if isinstance(model, NodeReference) else model
        self.model = model.copy()
        self.batch_size = batch_size

        # Prepare the computation_context
        seed = seed if not None else randomstate.entropy.random_entropy()
        self.model.computation_context.seed = seed
        self.model.computation_context.batch_size = self.batch_size

        self.client = Client

    def sample(self, n_samples, *args, **kwargs):
        """Run the sampler.

        Subsequent calls will reuse existing data without rerunning the
        simulator until necessary.

        Parameters
        ----------
        n_samples : int
            Number of samples from the posterior

        Returns
        -------
        A dictionary with at least the following items:
        samples : list of np.arrays
            Samples from the posterior distribution of each parameter.
        """
        raise NotImplementedError("Subclass implements")


class Rejection(InferenceMethod):
    """Rejection sampler.
    """

    def __init__(self, model, seed=None, batch_size=1000, discrepancy_name=None):
        """

        Parameters
        ----------
        model : ElfiModel or NodeReference
        seed : int
        batch_size : int

        """

        if isinstance(model, NodeReference):
            if isinstance(model, Discrepancy) and discrepancy_name is None:
                discrepancy_name = model.name

        super(Rejection, self).__init__(model, seed=seed, batch_size=batch_size)
        self.discrepancy_name = discrepancy_name

        self.compiled_net = self.client.compile(self.model,
                                                outputs=self.model.parameter_names +
                                                        [self.discrepancy_name])

    def sample(self, n_samples, quantile=0.01, threshold=None):
        """Run the rejection sampler.

        In quantile mode, the simulator is run (n/quantile) times.

        In threshold mode, the simulator is run until n_samples can be returned.
        Note that a threshold too small may result in a very long or never ending job.

        Parameters
        ----------
        n_samples : int
            Number of samples from the posterior
        quantile : float, optional
            The quantile in range ]0, 1] determines the acceptance threshold.
        threshold : float, optional
            The acceptance threshold.

        Returns
        -------
        result : dict

        Notes
        -----
        Currently the returned samples are ordered by the discrepancy

        """

        if quantile <= 0 or quantile > 1:
            raise ValueError("Quantile must be in range ]0, 1].")

        outputs = None

        # Quantile case
        if threshold is None:
            n_sim = int(np.ceil(n_samples / quantile))
            self.submit_in_batches(n_sim)

            while self.client.has_batches():
                batch_outputs, batch_index = self.wait_next_batch()

                # Initialize the outputs dict
                if outputs is None:
                    outputs = {}
                    for k, v in batch_outputs.items():
                        outputs[k] = np.ones((n_samples+self.batch_size,) + v.shape[1:])*np.inf

                # Put the acquired outputs to the end
                for k, v in outputs.items():
                    v[n_samples:] = batch_outputs[k]

                # Sort the smallest to the beginning
                sort_mask = np.argsort(outputs[self.discrepancy_name], axis=0).ravel()
                for k, v in outputs.items():
                    v[:] = v[sort_mask]

            # Because generation is in batches, the accurate accept_rate may deviate
            # slightly from quantile
            accept_rate = n_samples/n_sim
            threshold = outputs[self.discrepancy_name][n_samples-1]
            # Take the correct number of samples
            for k, v in outputs.items():
                outputs[k] = v[:n_samples]

        # Threshold case
        else:
            # TODO
            n_sim = self._ceil_in_batch_sizes(n_samples)
            while True:
                distances, parameters = self._acquire(n_sim)
                accepted_mask, n_accepted, accept_rate = self.compute_acceptance(distances,
                                                                            threshold)
                if n_accepted >= n_samples:
                    break

                n_sim = self.estimate_proposals_needed(n_samples, accept_rate)

        result = dict(outputs=outputs,
                      n_sim=n_sim,
                      accept_rate=accept_rate,
                      threshold=threshold)

        return result

    def submit_in_batches(self, n_sim):
        context = self.model.computation_context
        n_batches = int(np.ceil(n_sim/context.batch_size))
        batches = list(range(n_batches))
        self.client.submit_batches(batches, self.compiled_net, context)

    def wait_next_batch(self):
        return self.client.wait_next_batch()