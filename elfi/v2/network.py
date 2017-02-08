import uuid

from elfi.v2.executor import Executor

import networkx as nx


_current_network = None


def get_current_network():
    global _current_network
    if _current_network is None:
        _current_network = Network()
    return _current_network


class Network:
    def __init__(self):
        self._net = nx.DiGraph(name='default')

    def add_node(self, name, state):
        self._net.add_node(name, state)

    def get_node(self, name):
        return self._net.node[name]

    def add_edge(self, parent_name, child_name):
        self._net.add_edge(parent_name, child_name)

    def add_parent(self, name, parent):
        if not isinstance(parent, NodePointer):
            parent_name = "_{}_{}".format(name, str(uuid.uuid4().hex[0:6]))
            parent = Constant(parent_name, parent, network=self)

        self.add_edge(parent.name, name)

    def to_computation(self, outputs, n):
        """

        Parameters
        ----------
        outputs : list of node names
            Nodes whose output is computed
        span

        Returns
        -------

        """

        outputs = outputs if isinstance(outputs, list) else [outputs]

        # Filter out the unnecessary computations
        outputs_ancestors = set(outputs)
        for output in outputs:
            outputs_ancestors = outputs_ancestors.union(nx.ancestors(self._net, output))
        comp_net = self._net.subgraph(outputs_ancestors).copy()

        # Translate the nodes to computation nodes
        comp_net = OperationNetCompiler.compile(comp_net)

        # TODO: pass through Store objects that may alter the structure

        # Add variables
        comp_net.graph['n'] = n

        return comp_net

    def get_pointer(self, name):
        cls = self.get_node(name)['class']
        return cls.make(name, self)


# Computation graph compiler classes
class OperationNetCompiler:

    @staticmethod
    def compile(net):
        # Translate the nodes to computation nodes
        for name, d in net.nodes_iter(data=True):
            node_attr = d['class'].compile(d)
            net.node[name] = node_attr
        return net


class NodePointer:

    def __init__(self, name, *parents, state=None, network=None):
        state = state or {}
        state["class"] = self.__class__
        network = network or get_current_network()

        network.add_node(name, state)
        for p in parents:
            network.add_parent(name, p)

        self._init_pointer(name, network)

    @classmethod
    def make(cls, name, network):
        """Creates a pointer for an existing node

        Returns
        -------
        NodePointer instance
        """
        instance = cls.__new__(cls)
        instance._init_pointer(name, network)
        return instance

    def _init_pointer(self, name, network):
        """Initializes all internal variables of the NodePointer instance

        Parameters
        ----------
        name : name of the node in the network
        network : Network

        """
        self.name = name
        self.network = network

    def generate(self, n=1):
        comp_net = self.network.to_computation(self.name, n)
        result_net = Executor.execute(comp_net)
        return result_net.node[self.name]['output']

    def __getitem__(self, item):
        """

        Returns
        -------
        item from the state dict of the node
        """
        return self.network.get_node(self.name)[item]

    def __str__(self):
        return "{}('{}')".format(self.__class__.__name__, self.name)

    def __repr__(self):
        return self.__str__()


class Constant(NodePointer):
    def __init__(self, name, value, **kwargs):
        state = {
            "value": value,
        }
        super(Constant, self).__init__(name, state=state, **kwargs)

    @staticmethod
    def compile(state):
        return dict(output=state['value'])


""" TODO: below to be removed"""


class LegacyNetwork:
    """
    Currently we use networkX as our data layer. It can be pickled (assuming all the
    additional attributes are also pickleable).
    """

    # FIXME: using node for now in order to represent the ElfiModel. Switch to networkX
    #        based representation later
    def __init__(self, node):
        """
        """
        self.node = node

    def create_computation_network(self, outputs, batch_span):
        """

        Parameters
        ----------
        outputs : list of node names
            Nodes whose output is computed
        batch_span

        Returns
        -------

        """

        outputs = outputs if isinstance(outputs, list) else [outputs]

        # Turn the Elfi model to a networkx based computation graph
        comp_graph = self._inference_task_to_networkx(outputs, batch_span)

        # Filter out the unnecessary computations
        outputs_ancestors = set(outputs)
        for output in outputs:
            outputs_ancestors = outputs_ancestors.union(nx.ancestors(comp_graph, output))
        comp_graph = comp_graph.subgraph(outputs_ancestors)

        # TODO: pass through Store objects that may alter the structure

        return comp_graph

    # TODO: to be deprecated
    def _inference_task_to_networkx(self, outputs, batch_span):
        ancestors = self.node.ancestors
        comp_graph = nx.DiGraph(batch_span=batch_span)
        for ancestor in ancestors:
            print('Adding node {}'.format(ancestor.name))
            comp_graph.add_node(ancestor.name,
                                output=ancestor.transform,
                                observed=ancestor.observed if hasattr(ancestor,
                                                                      'observed') else None)
            print(comp_graph.node[ancestor.name]['observed'])
            for i, p in enumerate(ancestor.parents):
                comp_graph.add_edge(p.name, ancestor.name, pos=i)
        return comp_graph





