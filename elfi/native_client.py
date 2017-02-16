import logging

import networkx as nx

from elfi.executor import Executor
from elfi.utils import splen, all_ancestors, args_to_tuple, nx_search_iter


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

        # Use the constructor for a shallow copy (G.copy makes a deep copy in nx). This
        # causes e.g. the random_state objects of wrapped scipy distributions to be
        # copied and not use the original instance any longer.
        loaded_net = nx.DiGraph(compiled_net)

        # Add observed data
        loaded_net = ObservedLoader.load(model, loaded_net, span)
        loaded_net = BatchSizeLoader.load(model, loaded_net, span)
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


class Compiler:
    @classmethod
    def compile(cls, source_net, output_net):
        """

        Parameters
        ----------
        source_net : nx.DiGraph
        output_net : nx.DiGraph

        Returns
        -------

        """
        raise NotImplementedError


class OutputCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        logger.debug("OutputCompiler compiling...")

        # Make a structural copy of the source_net
        output_net.add_nodes_from(source_net.nodes())
        output_net.add_edges_from(source_net.edges(data=True))

        # Compile the nodes to computation nodes
        for name, data in output_net.nodes_iter(data=True):
            state = source_net.node[name]
            data['output'] = state['class'].compile_output(state)

        return output_net


class ObservedCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        """Adds observed nodes to the computation graph

        Parameters
        ----------
        source_net : nx.DiGraph
        output_net : nx.DiGraph

        Returns
        -------
        output_net : nx.DiGraph
        """
        logger.debug("ObservedCompiler compiling...")

        obs_net = nx.DiGraph(output_net)
        requires_observed = []

        for node, d in source_net.nodes_iter(data=True):
            if 'observed' in d.get('require', ()):
                requires_observed.append(node)
            elif not d.get('observable', False):
                obs_net.remove_node(node)

        renames = {k:cls.obs_name(k) for k in obs_net.nodes_iter()}
        nx.relabel_nodes(obs_net, renames, copy=False)

        output_net = nx.compose(output_net, obs_net)

        # Add the links to the nodes that require observed
        for node in requires_observed:
            obs_name = cls.obs_name(node)
            output_net.add_edge(obs_name, node, param='observed')
            # Combine the outputs
            output_net.node[obs_name]['output'] = args_to_tuple

        return output_net

    @staticmethod
    def obs_name(name):
        return "_{}_observed".format(name)


class RandomStateCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        for node, d in output_net.nodes(data=True):
            if 0:
                pass


class BatchSizeCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        logger.debug("BatchSizeCompiler compiling...")

        token = 'batch_size'
        _name = '_batch_size'
        for node, d in source_net.nodes_iter(data=True):
            if token in d.get('require', ()):
                if not output_net.has_node(_name):
                    output_net.add_node(_name)
                output_net.add_edge(_name, node, param=token)
        return output_net


class ReduceCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        logger.debug("ReduceCompiler compiling...")

        outputs = output_net.graph['outputs']
        output_ancestors = all_ancestors(output_net, outputs)
        for node in output_net.nodes():
            if node not in output_ancestors:
                output_net.remove_node(node)
        return output_net


class Loader:
    """
    Loads precomputed values to the compiled network
    """
    @classmethod
    def load(cls, model, output_net, span):
        """

        Parameters
        ----------
        model : ElfiModel
        output_net : nx.DiGraph
        span : tuple

        Returns
        -------

        """


class ObservedLoader(Loader):
    """
    Add observed data to computation graph
    """

    @classmethod
    def load(cls, model, output_net, span):
        for name, v in model.observed.items():
            obs_name = ObservedCompiler.obs_name(name)
            if not output_net.has_node(obs_name):
                continue
            output_net.node[obs_name] = dict(output=v)

        return output_net


class BatchSizeLoader(Loader):
    """
    Add observed data to computation graph
    """

    @classmethod
    def load(cls, model, output_net, span):
        _name = '_batch_size'
        if output_net.has_node(_name):
            output_net.node[_name]['output'] = splen(span)

        return output_net
