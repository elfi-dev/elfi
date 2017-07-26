import logging

import networkx as nx

from elfi.utils import args_to_tuple, nbunch_ancestors, observed_name


logger = logging.getLogger(__name__)


class Compiler:
    @classmethod
    def compile(cls, source_net, compiled_net):
        """

        Parameters
        ----------
        source_net : nx.DiGraph
        compiled_net : nx.DiGraph

        Returns
        -------
        compiled_net : nx.Digraph

        """
        raise NotImplementedError


class OutputCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, compiled_net):
        """Compiles the nodes present in the source_net
        """
        logger.debug("{} compiling...".format(cls.__name__))

        # Make a structural copy of the source_net
        compiled_net.add_nodes_from(source_net.nodes())
        compiled_net.add_edges_from(source_net.edges(data=True))

        # Compile the nodes to computation nodes
        for name, data in compiled_net.nodes_iter(data=True):
            state = source_net.node[name]
            if '_output' in state and '_operation' in state:
                raise ValueError("Cannot compile: both _output and _operation present "
                                 "for node '{}'".format(name))

            if '_output' in state:
                data['output'] = state['_output']
            elif '_operation' in state:
                data['operation'] = state['_operation']
            else:
                raise ValueError("Cannot compile, no _output or _operation present for "
                                 "node '{}'".format(name))

        return compiled_net


class ObservedCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, compiled_net):
        """Adds observed nodes to the computation graph
        """
        logger.debug("{} compiling...".format(cls.__name__))

        observable = []
        uses_observed = []

        for node in nx.topological_sort(source_net):
            state = source_net.node[node]
            if state.get('_observable'):
                observable.append(node)
                cls.make_observed_copy(node, compiled_net)
            elif state.get('_uses_observed'):
                uses_observed.append(node)
                obs_node = cls.make_observed_copy(node, compiled_net, args_to_tuple)
                # Make edge to the using node
                compiled_net.add_edge(obs_node, node, param='observed')
            else:
                continue

            # Copy the edges
            if not state.get('_stochastic'):
                obs_node = observed_name(node)
                for parent in source_net.predecessors(node):
                    if parent in observable:
                        link_parent = observed_name(parent)
                    else:
                        link_parent = parent

                    compiled_net.add_edge(link_parent, obs_node,
                                          source_net[parent][node].copy())

        # Check that there are no stochastic nodes in the ancestors
        for node in uses_observed:
            # Use the observed version to query observed ancestors in the compiled_net
            obs_node = observed_name(node)
            for ancestor_node in nx.ancestors(compiled_net, obs_node):
                if '_stochastic' in source_net.node.get(ancestor_node, {}):
                    raise ValueError("Observed nodes must be deterministic. Observed "
                                     "data depends on a non-deterministic node {}."
                                     .format(ancestor_node))

        return compiled_net

    @classmethod
    def make_observed_copy(cls, node, compiled_net, operation=None):
        obs_node = observed_name(node)

        if compiled_net.has_node(obs_node):
            raise ValueError("Observed node {} already exists!".format(obs_node))

        if operation is None:
            compiled_dict = compiled_net.node[node].copy()
        else:
            compiled_dict = dict(operation=operation)

        compiled_net.add_node(obs_node, compiled_dict)
        return obs_node


class AdditionalNodesCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, compiled_net):
        logger.debug("{} compiling...".format(cls.__name__))

        instruction_node_map = dict(_uses_batch_size='_batch_size',
                                    _uses_meta='_meta')

        for instruction, _node in instruction_node_map.items():
            for node, d in source_net.nodes_iter(data=True):
                if d.get(instruction):
                    if not compiled_net.has_node(_node):
                        compiled_net.add_node(_node)
                    compiled_net.add_edge(_node, node, param=_node[1:])

        return compiled_net


class RandomStateCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, compiled_net):
        logger.debug("{} compiling...".format(cls.__name__))

        _random_node = '_random_state'
        for node, d in source_net.nodes_iter(data=True):
            if '_stochastic' in d:
                if not compiled_net.has_node(_random_node):
                    compiled_net.add_node(_random_node)
                compiled_net.add_edge(_random_node, node, param='random_state')
        return compiled_net


class ReduceCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, compiled_net):
        logger.debug("{} compiling...".format(cls.__name__))

        outputs = compiled_net.graph['outputs']
        output_ancestors = nbunch_ancestors(compiled_net, outputs)
        for node in compiled_net.nodes():
            if node not in output_ancestors:
                compiled_net.remove_node(node)
        return compiled_net
