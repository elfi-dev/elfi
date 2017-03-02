import logging

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
        context = model.computation_context.copy()
        if seed is not None:
            context.seed = seed
        context.batch_size = self.batch_size
        self.model.computation_context = context
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

    @property
    def num_cores(self):
        return self.client.num_cores()


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

        self.compiled_net = self.client.compile(self.model.source_net,
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
            n_batches = int(np.ceil(n_samples/(quantile * self.batch_size)))
            self.submit_n_batches(n_batches)

            while self.has_batches():
                batch_outputs, batch_index = self.wait_next_batch()
                outputs = self.merge_batch(batch_outputs, outputs, n_samples)

        # Threshold case
        else:
            n_batches = self.num_cores
            self.submit_n_batches(n_batches)

            while self.has_batches():
                batch_outputs, batch_index = self.wait_next_batch()
                outputs = self.merge_batch(batch_outputs, outputs, n_samples)

                current_threshold = outputs[self.discrepancy_name][n_samples-1]
                if current_threshold > threshold:
                    # TODO: smarter rule for adding more computation
                    self.submit_n_batches(1, n_batches)
                    n_batches += 1
                else:
                    n_batches -= self.num_pending_batches()
                    self.client.clear_batches()
                    break

        # Prepare output
        n_sim = n_batches * self.batch_size
        accept_rate = n_samples / n_sim
        threshold = outputs[self.discrepancy_name][n_samples - 1]

        # Take out the correct number of samples
        for k, v in outputs.items():
            outputs[k] = v[:n_samples]

        result = dict(outputs=outputs,
                      n_sim=n_sim,
                      accept_rate=accept_rate,
                      threshold=threshold)

        return result

    def merge_batch(self, batch_outputs, outputs, n_samples):
        # TODO: add index vector so that you can recover the original order, also useful for async
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

        return outputs

    def submit_n_batches(self, n_batches, batch_offset=0):
        context = self.model.computation_context
        batches = list(range(batch_offset, batch_offset+n_batches))
        self.submit_batches(batches, self.compiled_net, context)

    def submit_batches(self, batches, compiled_net, context):
        self.client.submit_batches(batches, compiled_net, context)

    def wait_next_batch(self):
        return self.client.wait_next_batch()

    def has_batches(self):
        return self.client.has_batches()

    def num_pending_batches(self):
        return self.client.num_pending_batches(self.compiled_net,
                                               self.model.computation_context)