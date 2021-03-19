"""This module implements testbench-functionality to elfi"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['TestBench']

class TestBench:
    """Base class for comparing the performance of LFI-methods.

    Attributes
    ----------
    model : elfi.ElfiModel
        The ELFI graph used by the algorithm
    output_names : list
        Names of the nodes whose outputs are included in the batches
    client : elfi.client.ClientBase
        The batches are computed in the client
    max_parallel_batches : int
    state : dict
        Stores any changing data related to achieving the objective. Must include a key
        ``n_batches`` for determining when the inference is finished.
    objective : dict
        Holds the data for the algorithm to internally determine how many batches are still
        needed. You must have a key ``n_batches`` here. By default the algorithm finished when
        the ``n_batches`` in the state dictionary is equal or greater to the corresponding
        objective value.
    batches : elfi.client.BatchHandler
        Helper class for submitting batches to the client and keeping track of their
        indexes.
    pool : elfi.store.OutputPool
        Pool object for storing and reusing node outputs.


    """

    def __init__(self,
                 model,
                 output_names,
                 batch_size=1,
                 seed=None,
                 pool=None,
                 max_parallel_batches=None):
        """Construct the inference algorithm object.

        If you are implementing your own algorithm do not forget to call `super`.

        Parameters
        ----------
        model : ElfiModel
            Model to perform the inference with.
        output_names : list
            Names of the nodes whose outputs will be requested from the ELFI graph.
        batch_size : int, optional
            The number of parameter evaluations in each pass through the ELFI graph.
            When using a vectorized simulator, using a suitably large batch_size can provide
            a significant performance boost.
        seed : int, optional
            Seed for the data generation from the ElfiModel
        pool : OutputPool, optional
            OutputPool both stores and provides precomputed values for batches.
        max_parallel_batches : int, optional
            Maximum number of batches allowed to be in computation at the same time.
            Defaults to number of cores in the client


        """