"""This module contains the base client API and batch handler."""

import importlib
import logging
from collections import OrderedDict
from types import ModuleType

import networkx as nx

from elfi.compiler import (AdditionalNodesCompiler, ObservedCompiler,
                           OutputCompiler, RandomStateCompiler, ReduceCompiler)
from elfi.executor import Executor
from elfi.loader import AdditionalNodesLoader, ObservedLoader, PoolLoader, RandomStateLoader

logger = logging.getLogger(__name__)

_client = None  # a global for storing current client
_default_class = None  # a global for storing default client class


def get_client():
    """Get the current ELFI client instance."""
    global _client
    if _client is None:
        if _default_class is None:
            raise ValueError('Default client class is not defined')
        _client = _default_class()
    return _client


def set_client(client=None, **kwargs):
    """Set the current ELFI client instance.

    Parameters
    ----------
    client : ClientBase or str
        Instance of a client from ClientBase,
        or a string from ['native', 'multiprocessing', 'ipyparallel'].
        If string, the respective constructor is called with `kwargs`.

    """
    global _client

    if isinstance(client, str):
        m = importlib.import_module('elfi.clients.{}'.format(client))
        client = m.Client(**kwargs)

    _client = client


def set_default_class(class_or_module):
    """Set the default client class."""
    global _default_class
    if isinstance(class_or_module, ModuleType):
        class_or_module = class_or_module.Client
    _default_class = class_or_module


class BatchHandler:
    """Responsible for sending computational graphs to be executed in an Executor."""

    def __init__(self, model, context, output_names=None, client=None):
        """Compile the computational graph and associate it with a context etc.

        Parameters
        ----------
        model : ElfiModel
        context : ComputationContext
        output_names : list of str, optional
        client : Client, optional

        """
        client = client or get_client()

        self.compiled_net = client.compile(model.source_net, output_names)
        self.context = context
        self.client = client

        self._next_batch_index = 0
        self._pending_batches = OrderedDict()

    def has_ready(self, any=False):
        """Check if the next batch in succession is ready."""
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
        """Return the next batch index to be submitted."""
        return self._next_batch_index

    @property
    def total(self):
        """Return the total number of submitted batches."""
        return self._next_batch_index

    @property
    def num_ready(self):
        """Return the number of finished batches."""
        return self.total - self.num_pending

    @property
    def num_pending(self):
        """Return the total number of batches pending for evaluation."""
        return len(self.pending_indices)

    @property
    def has_pending(self):
        """Return whether any pending batches exist."""
        return self.num_pending > 0

    @property
    def pending_indices(self):
        """Return the keys to pending batches."""
        return self._pending_batches.keys()

    def cancel_pending(self):
        """Cancel all pending batches.

        Sets the next batch_index to the index of the last cancelled.

        Note that we rely here on the assumption that batches are processed in order.

        """
        for batch_index, id in reversed(list(self._pending_batches.items())):
            if batch_index != self._next_batch_index - 1:
                raise ValueError('Batches are not in order')

            logger.debug('Cancelling batch {}'.format(batch_index))
            self.client.remove_task(id)
            self._pending_batches.pop(batch_index)
            self._next_batch_index = batch_index

    def reset(self):
        """Cancel all pending batches and set the next index to 0."""
        self.cancel_pending()
        self._next_batch_index = 0

    def submit(self, batch=None):
        """Submit a batch with a batch index given by `next_index`.

        Parameters
        ----------
        batch : dict
            Overriding values for the batch.

        """
        batch = batch or {}
        batch_index = self._next_batch_index

        logger.debug('Submitting batch {}'.format(batch_index))
        loaded_net = self.client.load_data(self.compiled_net, self.context, batch_index)
        # Override
        for k, v in batch.items():
            loaded_net.nodes[k].update({'output': v})
            del loaded_net.nodes[k]['operation']

        task_id = self.client.submit(loaded_net)
        self._pending_batches[batch_index] = task_id

        # Update counters
        self._next_batch_index += 1
        self.context.num_submissions += 1

    def wait_next(self):
        """Wait for the next batch in succession."""
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
        """Return the number of processes."""
        return self.client.num_cores


class ClientBase:
    """Client api for serving multiple simultaneous inferences."""

    def apply(self, kallable, *args, **kwargs):
        """Add `kallable(*args, **kwargs)` to the queue of tasks and return immediately.

        Non-blocking apply.

        Parameters
        ----------
        kallable : callable
        args
            Positional arguments for the kallable
        kwargs
            Keyword arguments for the kallable

        Returns
        -------
        id : int
            Number of the queued task.

        """
        raise NotImplementedError

    def apply_sync(self, kallable, *args, **kwargs):
        """Call and returns the result of `kallable(*args, **kwargs)`.

        Blocking apply.

        Parameters
        ----------
        kallable : callable

        """
        raise NotImplementedError

    def get_result(self, task_id):
        """Return the result from task identified by `task_id` when it arrives.

        ELFI will call this only once per task_id.

        Parameters
        ----------
        task_id : int
            Id of the task whose result to return.

        """
        raise NotImplementedError

    def is_ready(self, task_id):
        """Return whether task with identifier `task_id` is ready.

        Parameters
        ----------
        task_id : int

        """
        raise NotImplementedError

    def remove_task(self, task_id):
        """Remove task with identifier `task_id` from pool.

        Parameters
        ----------
        task_id : int

        """
        raise NotImplementedError

    def reset(self):
        """Stop all worker processes immediately and clear pending tasks."""
        raise NotImplementedError

    def submit(self, loaded_net):
        """Add `loaded_net` to the queue of tasks and return immediately."""
        return self.apply(Executor.execute, loaded_net)

    def compute(self, loaded_net):
        """Request evaluation of `loaded_net` and wait for result."""
        return self.apply_sync(Executor.execute, loaded_net)

    @property
    def num_cores(self):
        """Return the number of processes."""
        raise NotImplementedError

    @classmethod
    def compile(cls, source_net, outputs=None):
        """Compile the structure of the output net.

        Does not insert any data into the net.

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
        if isinstance(outputs, list):
            outputs = set(outputs)
        elif isinstance(outputs, type(source_net.nodes())):
            outputs = outputs
        else:
            outputs = [outputs]

        compiled_net = nx.DiGraph(
            outputs=outputs, name=source_net.graph['name'], observed=source_net.graph['observed'])

        compiled_net = OutputCompiler.compile(source_net, compiled_net)
        compiled_net = ObservedCompiler.compile(source_net, compiled_net)
        compiled_net = AdditionalNodesCompiler.compile(source_net, compiled_net)
        compiled_net = RandomStateCompiler.compile(source_net, compiled_net)
        compiled_net = ReduceCompiler.compile(source_net, compiled_net)

        return compiled_net

    @classmethod
    def load_data(cls, compiled_net, context, batch_index):
        """Load data from the sources of the model and adds them to the compiled net.

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

        # Add cache from the contect
        loaded_net.graph['_executor_cache'] = context.caches['executor']

        return loaded_net
