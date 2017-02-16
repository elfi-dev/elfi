import logging

import networkx as nx

from elfi.utils import args_to_tuple, all_ancestors


logger = logging.getLogger(__name__)


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
        logger.debug("{} compiling...".format(cls.__name__))

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
        logger.debug("{} compiling...".format(cls.__name__))

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


class BatchSizeCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        logger.debug("{} compiling...".format(cls.__name__))

        token = 'batch_size'
        _name = '_batch_size'
        for node, d in source_net.nodes_iter(data=True):
            if token in d.get('require', ()):
                if not output_net.has_node(_name):
                    output_net.add_node(_name)
                output_net.add_edge(_name, node, param=token)
        return output_net


class RandomStateCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        logger.debug("{} compiling...".format(cls.__name__))

        token = 'stochastic'
        _name = '_{}_random_state'
        for node, d in source_net.nodes_iter(data=True):
            if token in d:
                random_node = _name.format(node)
                output_net.add_node(random_node)
                output_net.add_edge(random_node, node, param='random_state')
        return output_net


class ReduceCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        logger.debug("{} compiling...".format(cls.__name__))

        outputs = output_net.graph['outputs']
        output_ancestors = all_ancestors(output_net, outputs)
        for node in output_net.nodes():
            if node not in output_ancestors:
                output_net.remove_node(node)
        return output_net