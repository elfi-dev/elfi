import networkx as nx
from operator import itemgetter


class GraphicalModel:
    """
    Network class for the ElfiModel.
    """
    def __init__(self, source_net=None):
        self.source_net = source_net or nx.DiGraph()

    def add_node(self, name, state):
        if self.has_node(name):
            raise ValueError('Node {} already exists'.format(name))
        self.source_net.add_node(name, attr_dict=state)

    def remove_node(self, name):
        self.source_net.remove_node(name)

    def get_node(self, name):
        """Returns the state of the node

        Returns
        -------
        out : dict
        """
        return self.source_net.node[name]

    def set_node(self, name, state):
        """Set the state of the node"""
        self.source_net.node[name] = state

    def has_node(self, name):
        return self.source_net.has_node(name)

    # TODO: deprecated. Incorporate into add_node so that these are not modifiable
    # This protects the internal state of the ElfiModel so that consistency can be more
    # easily managed
    def add_edge(self, parent_name, child_name, param_name=None):
        # Deprecated. By default, map to a positional parameter of the child
        if param_name is None:
            param_name = len(self.parent_names(child_name))
        if not isinstance(param_name, (int, str)):
            raise ValueError('Unrecognized type for `param_name` {}. Must be either an '
                             '`int` for positional parameters or `str` for named '
                             'parameters.'.format(param_name))

        if not self.has_node(parent_name):
            raise ValueError('Parent {} does not exist'.format(parent_name))
        if not self.has_node(child_name):
            raise ValueError('Child {} does not exist'.format(child_name))

        self.source_net.add_edge(parent_name, child_name, param=param_name)

    def replace_node(self, node, replacing_node):
        """Replaces the node such that node's state and parents are redefined. Children
        are preserved.

        Parameters
        ----------
        node : str
        replacing_node : str
        """

        # Clear current incoming edges
        for u,v in self.source_net.in_edges(node):
            self.source_net.remove_edge(u, v)

        # Transfer incoming edges
        for u, v, data in self.source_net.in_edges(replacing_node, data=True):
            self.source_net.add_edge(u, node, data)

        # Transfer state and remove replacing node from the graph
        self.source_net.node[node] = self.source_net.node[replacing_node]
        self.source_net.remove_node(replacing_node)

    def parent_names(self, child_name):
        """

        Parameters
        ----------
        child_name

        Returns
        -------
        parent_names : list
            List of positional parent names
        """
        args = []
        for parent_name in self.source_net.predecessors(child_name):
            param = self.source_net[parent_name][child_name]['param']
            if isinstance(param, int):
                args.append((param, parent_name))
        return [a[1] for a in sorted(args, key=itemgetter(0))]

    @property
    def nodes(self, data=False):
        return self.source_net.nodes(data)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        kopy = self.__class__()
        kopy.source_net = nx.DiGraph(self.source_net)
        return kopy