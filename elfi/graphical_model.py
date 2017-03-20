import networkx as nx


class GraphicalModel:
    """
    Network class for the ElfiModel.
    """
    def __init__(self, source_net=None):
        self.source_net = source_net or nx.DiGraph()

    def add_node(self, name, state):
        if self.source_net.has_node(name):
            raise ValueError('Node {} already exists'.format(name))
        self.source_net.add_node(name, attr_dict=state)

    def get_node(self, name):
        """Returns the state of the node

        Returns
        -------
        out : dict
        """
        return self.source_net.node[name]

    def add_edge(self, parent_name, child_name, param=None):
        # By default, map to a positional parameter of the child
        if param is None:
            param = len(self.source_net.predecessors(child_name))

        if not self.source_net.has_node(parent_name):
            raise ValueError('Parent {} does not exist'.format(parent_name))
        if not self.source_net.has_node(child_name):
            raise ValueError('Child {} does not exist'.format(child_name))

        self.source_net.add_edge(parent_name, child_name, param=param)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        kopy = self.__class__()
        kopy.source_net = nx.DiGraph(self.source_net)
        return kopy