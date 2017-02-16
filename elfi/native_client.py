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

    @classmethod
    def generate(cls, model, n, outputs, with_values=None):
        compiled_net = cls.compile(model, outputs)
        loaded_net = cls.load_data(model, compiled_net, (0, n))
        result = cls.execute(loaded_net, override_outputs=with_values)
        return result

    @classmethod
    def compile(cls, model, outputs):
        """Compiles the structure of the output net. Does not insert any data
        into the net.

        Parameters
        ----------
        model : ElfiModel
        outputs : list of node names

        Returns
        -------
        output_net : nx.DiGraph
            output_net codes the execution of the model
        """
        source_net = model._net
        outputs = outputs if isinstance(outputs, list) else [outputs]
        compiled_net = nx.DiGraph(outputs=outputs)

        compiled_net = OutputCompiler.compile(source_net, compiled_net)
        compiled_net = ObservedCompiler.compile(source_net, compiled_net)
        compiled_net = BatchSizeCompiler.compile(source_net, compiled_net)
        compiled_net = RandomStateCompiler.compile(source_net, compiled_net)
        compiled_net = ReduceCompiler.compile(source_net, compiled_net)


        return compiled_net

    @classmethod
    def load_data(cls, model, compiled_net, span):
        """Loads data from the sources of the model and adds them to the compiled net.

        Parameters
        ----------
        model : ElfiModel
        compiled_net : nx.DiGraph
        span : tuple
           (start index, end_index)
        values : dict
           additional values to be inserted into the network

        Returns
        -------
        output_net : nx.DiGraph
        """

        # Make a shallow copy of the graph
        loaded_net = nx.DiGraph(compiled_net)

        loaded_net = ObservedLoader.load(model, loaded_net, span)
        loaded_net = BatchSizeLoader.load(model, loaded_net, span)
        loaded_net = RandomStateLoader.load(model, loaded_net, span)
        # TODO: Add saved data from stores

        return loaded_net

    @classmethod
    def execute(cls, loaded_net, override_outputs=None):
        """Execute the computational graph"""

        loaded_net = cls._override_outputs(loaded_net, override_outputs)
        return Executor.execute(loaded_net)

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
