from graphviz import Digraph
import abcpy.core as core


def draw_model(discrepancy_node, draw_constants=False):
    """
    Return a GraphViz representation of the model.

    Inputs:
    - discrepancy_node: final node in the model.
    - draw_constants: whether to include Constant nodes
    """

    # gather the set of nodes, excluding Constants
    nodes = discrepancy_node.component

    if not draw_constants:
        nodes = [n for n in nodes if not isinstance(n, core.Constant)]

    default = {'shape': 'box', 'fillcolor': 'grey', 'style': 'solid'}
    dot = Digraph()

    # add nodes to graph
    for n in nodes:
        dot.node(n.name, **default)

        # TODO: different styles
        # if hasattr(n, 'observed'):
        #     dot.node(n.name, **default)
        # elif isinstance(n, core.Threshold):
        #     dot.node(n.name, **default)
        # elif isinstance(n, core.Discrepancy):
        #     dot.node(n.name, **default)
        # elif isinstance(n, core.Simulator):
        #     dot.node(n.name, **default)
        # elif isinstance(n, core.Value):
        #     dot.node(n.name, shape='point', xlabel=n.name)
        # else:
        #     dot.node(n.name, shape='doublecircle',
        #            fillcolor='deepskyblue3',
        #            style='filled')

    # add edges to graph
    edges = []
    for n in nodes:
        for c in n.children:
            if (n.name, c.name) not in edges:
                edges.append((n.name, c.name))
                dot.edge(n.name, c.name)
        for p in n.parents:
            if draw_constants or not isinstance(p, core.Constant):
                if (p.name, n.name) not in edges:
                    edges.append((p.name, n.name))
                    dot.edge(p.name, n.name)

    return dot
