from operator import itemgetter

import networkx as nx


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
        parent_names = self.get_parents(name)
        self.source_net.remove_node(name)

        # Remove sole private parents
        for p in parent_names:
            if p[0] == '_' and len(self.source_net.successors(p)) == 0 \
                    and len(self.source_net.predecessors(p)) == 0:
                self.remove_node(p)

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
            param_name = len(self.get_parents(child_name))
        if not isinstance(param_name, (int, str)):
            raise ValueError('Unrecognized type for `param_name` {}. Must be either an '
                             '`int` for positional parameters or `str` for named '
                             'parameters.'.format(param_name))

        if not self.has_node(parent_name):
            raise ValueError('Parent {} does not exist'.format(parent_name))
        if not self.has_node(child_name):
            raise ValueError('Child {} does not exist'.format(child_name))

        self.source_net.add_edge(parent_name, child_name, param=param_name)

    def update_node(self, node, updating_node):
        """Updates `node` with `updating_node` in the model.
         
        Node `node` gets the state (operation) and parents of the `updating_node`. The
        updating node is then removed from the graph.

        Parameters
        ----------
        node : str
        updating_node : str
        """

        out_edges = self.source_net.out_edges(node, data=True)
        self.remove_node(node)
        self.source_net.add_node(node, self.source_net.node[updating_node])
        self.source_net.add_edges_from(out_edges)

        # Transfer incoming edges
        for u, v, data in self.source_net.in_edges(updating_node, data=True):
            self.source_net.add_edge(u, node, data)

        self.remove_node(updating_node)

    def get_parents(self, child_name):
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
    def nodes(self):
        """Returns a list of nodes"""
        return self.source_net.nodes()

    def copy(self):
        kopy = self.__class__()
        # Copy the source net
        kopy.source_net = nx.DiGraph(self.source_net)
        return kopy

    def __copy__(self, *args, **kwargs):
        return self.copy()
