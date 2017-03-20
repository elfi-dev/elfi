import logging
from types import ModuleType

import networkx as nx

from elfi.compiler import OutputCompiler, ObservedCompiler, BatchSizeCompiler, \
    ReduceCompiler, RandomStateCompiler
from elfi.loader import ObservedLoader, BatchSizeLoader, RandomStateLoader, \
    OutputSupplyLoader, PoolLoader

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


class ClientBase:
    """
    Responsible for sending computational graphs to be executed in an Executor
    """

    def clear_batches(self):
        raise NotImplementedError

    def execute(self, loaded_net):
        raise NotImplementedError

    def has_batches(self):
        raise NotImplementedError

    def num_pending_batches(self, compiled_net=None, context=None):
        raise NotImplementedError

    def num_cores(self):
        raise NotImplementedError

    def submit_batches(self, batches, compiled_net, context):
        raise NotImplementedError

    def wait_next_batch(self, async=False):
        raise NotImplementedError

    def compile(self, source_net, outputs):
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
        outputs = outputs if isinstance(outputs, list) else [outputs]
        compiled_net = nx.DiGraph(outputs=outputs)

        compiled_net = OutputCompiler.compile(source_net, compiled_net)
        compiled_net = ObservedCompiler.compile(source_net, compiled_net)
        compiled_net = BatchSizeCompiler.compile(source_net, compiled_net)
        compiled_net = RandomStateCompiler.compile(source_net, compiled_net)
        compiled_net = ReduceCompiler.compile(source_net, compiled_net)

        return compiled_net

    def compute_batch(self, model, outputs, batch_index=0, context=None):
        """Blocking call to compute a batch from the model."""

        context = context or model.computation_context
        compiled_net = self.compile(model.source_net, outputs)
        loaded_net = self.load_data(compiled_net, context, batch_index)
        return self.execute(loaded_net)

    def load_data(self, compiled_net, context, batch_index):
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
        loaded_net = BatchSizeLoader.load(context, loaded_net, batch_index)
        loaded_net = RandomStateLoader.load(context, loaded_net, batch_index)
        loaded_net = OutputSupplyLoader.load(context, loaded_net, batch_index)
        loaded_net = PoolLoader.load(context, loaded_net, batch_index)
        # TODO: Add saved data from stores

        return loaded_net
