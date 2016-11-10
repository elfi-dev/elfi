from graphviz import Digraph
import elfi.core as core


def draw_model(discrepancy_node, draw_constants=False, filename=None):
    """
    Return a GraphViz dot representation of the model.

    Parameters
    ----------
    discrepancy_node : Node
        Final node in the model.
    draw_constants : boolean, optional
        Whether to draw Constant nodes.
    filename : string, optional
        If given, save the dot file into the given filename, trying to guess the type.
        For example: 'mymodel.png'.
    """

    # gather the set of nodes, excluding Constants
    nodes = discrepancy_node.component

    if not draw_constants:
        nodes = [n for n in nodes if not isinstance(n, core.Constant)]

    dot = Digraph()

    # add nodes to graph
    for n in nodes:
        node_format = {'shape': 'circle', 'fillcolor': 'grey', 'style': 'solid'}

        if hasattr(n, 'observed'):
            node_format['style'] = 'filled'

        dot.node(n.name, **node_format)

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

    if filename is not None:
        try:
            filebase, filetype = filename.split('.')
            dot.format = filetype
            dot.render(filebase)
        except:
            raise ValueError('Problem with the given filename.')

    return dot
