import networkx as nx


class GraphicalModel:
    """
    Network class for the ElfiModel.
    """
    def __init__(self):
        self._net = nx.DiGraph(name='default')

    def add_node(self, name, state):
        if self._net.has_node(name):
            raise ValueError('Node {} already exists'.format(name))
        self._net.add_node(name, attr_dict=state)

    def get_node(self, name):
        """Returns the state of the node

        Returns
        -------
        out : dict
        """
        return self._net.node[name]

    def add_edge(self, parent_name, child_name, param=None):
        # By default, map to a positional parameter of the child
        if param is None:
            param = len(self._net.predecessors(child_name))

        if not self._net.has_node(parent_name):
            raise ValueError('Parent {} does not exist'.format(parent_name))
        if not self._net.has_node(child_name):
            raise ValueError('Child {} does not exist'.format(child_name))

        self._net.add_edge(parent_name, child_name, param=param)

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        copy = self.__class__()
        copy._net = nx.DiGraph(self._net)
        return copy