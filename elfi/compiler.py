import logging

import networkx as nx

from elfi.utils import args_to_tuple, all_ancestors, observed_name


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

        observable = []
        uses_observed = []

        for node in nx.topological_sort(source_net):
            state = source_net.node[node]
            if state.get('observable'):
                observable.append(node)
                cls.make_observed_copy(node, output_net)
            elif state.get('uses_observed'):
                uses_observed.append(node)
                obs_node = cls.make_observed_copy(node, output_net, args_to_tuple)
                # Make edge to the using node
                output_net.add_edge(obs_node, node, param='observed')
            else:
                continue

            # Copy the edges
            if not state.get('stochastic'):
                obs_node = observed_name(node)
                for parent in source_net.predecessors(node):
                    if parent in observable:
                        link_parent = observed_name(parent)
                    else:
                        link_parent = parent

                    output_net.add_edge(link_parent, obs_node, source_net[parent][node].copy())

        # Check that there are no stochastic nodes in the ancestors
        # TODO: move to loading phase when checking that stochastic observables get their data?
        for node in uses_observed:
            # Use the observed version to query observed ancestors in the output_net
            obs_node = observed_name(node)
            for ancestor_node in nx.ancestors(output_net, obs_node):
                if 'stochastic' in source_net.node.get(ancestor_node, {}):
                    raise ValueError("Observed nodes must be deterministic. Observed data"
                                     "depends on a non-deterministic node {}."
                                     .format(ancestor_node))

        return output_net

    @classmethod
    def make_observed_copy(cls, node, output_net, output_data=None):
        obs_node = observed_name(node)

        if output_net.has_node(obs_node):
            raise ValueError("Observed node {} already exists!".format(obs_node))

        if output_data is None:
            output_dict = output_net.node[node].copy()
        else:
            output_dict = dict(output=output_data)

        output_net.add_node(obs_node, output_dict)
        return obs_node


class BatchSizeCompiler(Compiler):
    @classmethod
    def compile(cls, source_net, output_net):
        logger.debug("{} compiling...".format(cls.__name__))

        token = 'batch_size'
        _name = '_batch_size'
        for node, d in source_net.nodes_iter(data=True):
            if d.get('uses_batch_size'):
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