import logging
import importlib
from types import ModuleType
from collections import OrderedDict

import networkx as nx

from elfi.executor import Executor
from elfi.compiler import OutputCompiler, ObservedCompiler, AdditionalNodesCompiler, \
    ReduceCompiler, RandomStateCompiler
from elfi.loader import ObservedLoader, AdditionalNodesLoader, RandomStateLoader, \
    PoolLoader

logger = logging.getLogger(__name__)


_client = None
_default_class = None


def get_client():
    """Get the current ELFI client instance."""
    global _client
    if _client is None:
        if _default_class is None:
            raise ValueError('Default client class is not defined')
        _client = _default_class()
    return _client


def set_client(client=None):
    """Set the current ELFI client instance."""
    global _client

    if isinstance(client, str):
        m = importlib.import_module('elfi.clients.{}'.format(client))
        client = m.Client()

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

    def __init__(self, model, context, output_names=None, client=None):
        client = client or get_client()

        self.compiled_net = client.compile(model.source_net, output_names)
        self.context = context
        self.client = client

        self._next_batch_index = 0
        self._pending_batches = OrderedDict()

    def has_ready(self, any=False):
        """Check if the next batch in succession is ready"""
        if len(self._pending_batches) == 0:
            return False

        for bi, id in self._pending_batches.items():
            if self.client.is_ready(id):
                return True
            if not any:
                break
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

    def cancel_pending(self):
        """Cancels all the pending batches and sets the next batch_index to the index of
        the last cancelled.

        Note that we rely here on the assumption that batches are processed in order.

        Returns
        -------

        """
        for batch_index, id in reversed(list(self._pending_batches.items())):
            if batch_index != self._next_batch_index - 1:
                raise ValueError('Batches are not in order')

            logger.debug('Cancelling batch {}'.format(batch_index))
            self.client.remove_task(id)
            self._pending_batches.pop(batch_index)
            self._next_batch_index = batch_index

    def reset(self):
        """Cancels all the pending batches and sets the next index to 0
        """
        self.cancel_pending()
        self._next_batch_index = 0

    def submit(self, batch=None):
        """Submits a batch with a batch index given by `next_index`.

        Parameters
        ----------
        batch : dict
            Overriding values for the batch.

        Returns
        -------

        """
        batch = batch or {}
        batch_index = self._next_batch_index

        logger.debug('Submitting batch {}'.format(batch_index))
        loaded_net = self.client.load_data(self.compiled_net, self.context, batch_index)
        # Override
        for k,v in batch.items(): loaded_net.node[k] = {'output': v}

        task_id = self.client.submit(loaded_net)
        self._pending_batches[batch_index] = task_id

        # Update counters
        self._next_batch_index += 1
        self.context.num_submissions += 1

    def wait_next(self):
        """Waits for the next batch in succession"""
        if len(self._pending_batches) == 0:
            raise ValueError('Cannot wait for a batch, no batches currently submitted')

        batch_index, task_id = self._pending_batches.popitem(last=False)
        batch = self.client.get_result(task_id)
        logger.debug('Received batch {}'.format(batch_index))

        self.context.callback(batch, batch_index)
        return batch, batch_index

    def compute(self, batch_index=0):
        """Blocking call to compute a batch from the model."""
        loaded_net = self.client.load_data(self.compiled_net, self.context, batch_index)
        return self.client.compute(loaded_net)

    @property
    def num_cores(self):
        return self.client.num_cores


class ClientBase:
    """Client api for serving multiple simultaneous inferences"""

    def apply(self, kallable, *args, **kwargs):
        """Non-blocking apply, returns immediately with an id for the task.

        Parameters
        ----------
        kallable : callable
        args
            Positional arguments for the kallable
        kwargs
            Keyword arguments for the kallable

        """
        raise NotImplementedError

    def apply_sync(self, kallable, *args, **kwargs):
        """Blocking apply, returns the result."""
        raise NotImplementedError

    def get_result(self, task_id):
        """Get the result of the task.

        ELFI will call this only once per task_id."""
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

        compiled_net = nx.DiGraph(outputs=outputs, name=source_net.graph['name'],
                                  observed=source_net.graph['observed'])

        compiled_net = OutputCompiler.compile(source_net, compiled_net)
        compiled_net = ObservedCompiler.compile(source_net, compiled_net)
        compiled_net = AdditionalNodesCompiler.compile(source_net, compiled_net)
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
        loaded_net = AdditionalNodesLoader.load(context, loaded_net, batch_index)
        loaded_net = RandomStateLoader.load(context, loaded_net, batch_index)
        loaded_net = PoolLoader.load(context, loaded_net, batch_index)

        return loaded_net
