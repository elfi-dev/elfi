import logging

import networkx as nx

from elfi.compiler import OutputCompiler, ObservedCompiler, BatchSizeCompiler, \
    ReduceCompiler, RandomStateCompiler
from elfi.executor import Executor
from elfi.loader import ObservedLoader, BatchSizeLoader, RandomStateLoader

logger = logging.getLogger(__name__)


class Client:
    """
    Responsible for sending computational graphs to be executed in an Executor
    """

    submit_queue = list()

    @classmethod
    def submit_batches(cls, batches, compiled_net, context):
        cls.submit_queue.append((batches, compiled_net, context))

    @classmethod
    def has_batches(cls):
        return len(cls.submit_queue) > 0

    @classmethod
    def wait_next_batch(cls):
        batches, compiled_net, context = cls.submit_queue.pop(0)
        batch_index = batches.pop(0)

        batch_net = cls.load_data(context, compiled_net, batch_index)

        # Insert back to queue if batches left
        if len(batches) > 0:
            submitted = (batches, compiled_net, context)
            cls.submit_queue.insert(0, submitted)

        outputs = cls.execute(batch_net)
        return outputs, batch_index

    @classmethod
    def compute_batch(cls, model, outputs, batch_index=0, context=None):
        """Blocking call to compute a batch from the model."""

        context = context or model.computation_context
        compiled_net = cls.compile(model.source_net, outputs)
        loaded_net = cls.load_data(context, compiled_net, batch_index)
        return cls.execute(loaded_net, override_outputs=context.override_outputs)

    @classmethod
    def compile(cls, source_net, outputs):
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

    @classmethod
    def load_data(cls, context, compiled_net, batch_index):
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
        # TODO: Add saved data from stores

        return loaded_net

    # TODO: move override_outputs to ComputationContext and to the loading phase
    @classmethod
    def execute(cls, loaded_net, override_outputs=None):
        """Execute the computational graph"""

        loaded_net = cls._override_outputs(loaded_net, override_outputs)
        return Executor.execute(loaded_net)

    @classmethod
    def num_cores(cls):
        return 1

    @classmethod
    def num_pending_batches(cls, compiled_net=None, context=None):
        n = 0
        for submitted in cls.submit_queue:
            if compiled_net and compiled_net != submitted[1]:
                continue
            elif context and context != submitted[2]:
                continue
            n += len(submitted[0])
        return n

    @classmethod
    def clear_batches(cls):
        del cls.submit_queue[:]

    @classmethod
    def _override_outputs(cls, loaded_net, outputs):
        """

        Parameters
        ----------
        loaded_net : nx.DiGraph
        outputs : dict

        Returns
        -------

        """
        outputs = outputs or {}
        for name, v in outputs.items():
            if not loaded_net.node.get(name, False):
                raise ValueError("Node {} not found.".format(name))
            out_edges = loaded_net.out_edges(name, data=True)
            loaded_net.remove_node(name)
            loaded_net.add_node(name, output=v)
            loaded_net.add_edges_from(out_edges)
        return loaded_net
