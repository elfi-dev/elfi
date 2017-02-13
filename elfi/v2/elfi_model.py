import networkx as nx


_current_model = None


def get_current_model():
    global _current_model
    if _current_model is None:
        _current_model = ElfiModel()
    return _current_model


class ElfiModel:
    def __init__(self):
        self._net = nx.DiGraph(name='default')
        self.observed = {}

    def add_node(self, name, state):
        self._net.add_node(name, attr_dict=state)

    def get_node(self, name):
        return self._net.node[name]

    def add_edge(self, parent_name, child_name, param=None):
        # By default, map to a positional parameter of the child
        if param is None:
            param = len(self._net.predecessors(child_name))

        self._net.add_edge(parent_name, child_name, param=param)

    def get_reference(self, name):
        cls = self.get_node(name)['class']
        return cls.reference(name, self)
