import logging
from types import ModuleType
from collections import OrderedDict

import networkx as nx

from elfi.executor import Executor
from elfi.compiler import OutputCompiler, ObservedCompiler, BatchMetaCompiler, \
    ReduceCompiler, RandomStateCompiler
from elfi.loader import ObservedLoader, BatchMetaLoader, RandomStateLoader, PoolLoader
from elfi.store import OutputPool

logger = logging.getLogger(__name__)


_client = None
_default_class = None


def get():
    global _client
    if _client is None:
        if _default_class is None:
            raise ValueError('Default client class is not defined')
        _client = _default_class()
    return _client


def reset_default(client=None):
    global _client
    _client = client


def set_default_class(class_or_module):
    global _default_class
    if isinstance(class_or_module, ModuleType):
        class_or_module = class_or_module.Client
    _default_class = class_or_module


class BatchHandler:
    """
    Responsible for sending computational graphs to be executed in an Executor
    """

    def __init__(self, model, outputs=None, client=None):
        self.client = client or get()
        self.compiled_net = self.client.compile(model.source_net, outputs)
        self.context = model.computation_context

        self._next_batch_index = 0
        self._pending_batches = OrderedDict()

    @property
    def has_ready(self, batch_index=None):
        for bi, id in self._pending_batches.items():
            if batch_index and batch_index != bi:
                continue
            if self.client.is_ready(id):
                return True
        return False

    @property
    def next_index(self):
        """Returns the next batch index to be submitted"""
        return self._next_batch_index

    @property
    def total(self):
        return self._next_batch_index

    @property
    def num_ready(self):
        return self.total - self.num_pending

    @property
    def num_pending(self):
        return len(self.pending_indices)

    @property
    def has_pending(self):
        return self.num_pending > 0

    @property
    def pending_indices(self):
        return self._pending_batches.keys()

    def cancel_pending(self, offset=0):
        """Cancels all the pending batches equal to or above `offset` (default 0)

        Parameters
        ----------
        offset : int

        Returns
        -------

        """
        for batch_index, id in list(self._pending_batches.items()):
            if batch_index < offset:
                continue

            logger.debug('Cancelling batch {}'.format(batch_index))
            self.client.remove_task(id)
            self._pending_batches.pop(batch_index)

    def reset(self, offset=0):
        """Cancels all the pending batches equal to or above `offset` (default 0) and
        sets the next_batch_index to offset.

        Parameters
        ----------
        offset : int
            the index at which onward to reset the pending batches

        Returns
        -------

        """
        if offset < 0:
            raise ValueError('offset must be at least 0')
        self.cancel_pending(offset)

        self._next_batch_index = offset

    def submit(self, batch=None):
        """

        Parameters
        ----------
        batch : dict
            Overriding values for the batch. Will be added to the pool.

        Returns
        -------

        """
        batch_index = self._next_batch_index
        if batch:
            # TODO: what if data already exists?
            self._add_to_pool(batch, batch_index)

        logger.debug('Submitting batch {}'.format(batch_index))
        loaded_net = self.client.load_data(self.compiled_net, self.context, batch_index)
        task_id = self.client.submit(loaded_net)
        self._pending_batches[batch_index] = task_id

        self._next_batch_index += 1

    def wait_next(self):
        """Waits for the next batch in succession"""
        batch_index, task_id = self._pending_batches.popitem(last=False)
        batch = self.client.get(task_id)
        logger.debug('Received batch {}'.format(batch_index))

        self.context.callback(batch, batch_index)
        return batch, batch_index

    def compute(self, batch_index=0):
        """Blocking call to compute a batch from the model."""
        loaded_net = self.client.load_data(self.compiled_net, self.context, batch_index)
        return self.client.compute(loaded_net)

    def _add_to_pool(self, batch, batch_index):
        if not self.context.pool:
            self.context.pool = OutputPool()
        pool = self.context.pool

        # Create missing stores
        for node in batch:
            if node not in pool:
                pool.add_store(node, {})

        pool.add_batch(batch, batch_index)

    @property
    def num_cores(self):
        return self.client.num_cores


class ClientBase:
    """Client api for serving multiple simultaneous inferences"""

    # TODO: add the self.tasks dict available
    # TODO: test that client is emptied from tasks as they are received

    def apply(self, kallable, *args, **kwargs):
        """Returns immediately with an id for the task"""
        raise NotImplementedError

    def apply_sync(self, kallable, *args, **kwargs):
        """Returns the result"""
        raise NotImplementedError

    def get(self, task_id):
        raise NotImplementedError

    def wait_next(self, task_ids):
        raise NotImplementedError

    def is_ready(self, task_id):
        """Queries whether task with id is completed"""
        raise NotImplementedError

    def remove_task(self, task_id):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def submit(self, loaded_net):
        return self.apply(Executor.execute, loaded_net)

    def compute(self, loaded_net):
        return self.apply_sync(Executor.execute, loaded_net)

    @property
    def num_cores(self):
        raise NotImplementedError

    @classmethod
    def compile(cls, source_net, outputs=None):
        """Compiles the structure of the output net. Does not insert any data
        into the net.

        Parameters
        ----------
        source_net : nx.DiGraph
            Can be acquired from `model.source_net`
        outputs : list of node names

        Returns
        -------
        output_net : nx.DiGraph
            output_net codes the execution of the model
        """
        if outputs is None:
            outputs = source_net.nodes()
        if not outputs:
            logger.warning("Compiling for no outputs!")
        outputs = outputs if isinstance(outputs, list) else [outputs]
        compiled_net = nx.DiGraph(outputs=outputs)

        compiled_net = OutputCompiler.compile(source_net, compiled_net)
        compiled_net = ObservedCompiler.compile(source_net, compiled_net)
        compiled_net = BatchMetaCompiler.compile(source_net, compiled_net)
        compiled_net = RandomStateCompiler.compile(source_net, compiled_net)
        compiled_net = ReduceCompiler.compile(source_net, compiled_net)

        return compiled_net

    @classmethod
    def load_data(cls, compiled_net, context, batch_index):
        """Loads data from the sources of the model and adds them to the compiled net.

        Parameters
        ----------
        context : ComputationContext
        compiled_net : nx.DiGraph
        batch_index : int

        Returns
        -------
        output_net : nx.DiGraph
        """

        # Make a shallow copy of the graph
        loaded_net = nx.DiGraph(compiled_net)

        loaded_net = ObservedLoader.load(context, loaded_net, batch_index)
        loaded_net = BatchMetaLoader.load(context, loaded_net, batch_index)
        loaded_net = RandomStateLoader.load(context, loaded_net, batch_index)
        loaded_net = PoolLoader.load(context, loaded_net, batch_index)

        return loaded_net
