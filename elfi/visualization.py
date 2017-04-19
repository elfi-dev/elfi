from elfi.model.elfi_model import ElfiModel, NodeReference


def nx_draw(G, internal=False, param_names=False, filename=None):
    """
    Return a GraphViz dot representation of the model.

    Requires the optional 'graphviz' library.

    Parameters
    ----------
    G : nx.DiGraph or ElfiModel
        Graph or model to draw
    internal : boolean, optional
        Whether to draw internal nodes (starting with an underscore)
    filename : string, optional
        If given, save the dot file into the given filename, trying to guess the type.
        For example: 'mymodel.png'.
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("The graphviz library is required for this feature.")

    if isinstance(G, ElfiModel):
        G = G.source_net
    elif isinstance(G, NodeReference):
        G = G.model.source_net

    dot = Digraph()

    for n, state in G.nodes_iter(data=True):
        if not internal and n[0] == '_':
            continue
        _format = {'shape': 'circle', 'fillcolor': 'gray80', 'style': 'solid'}
        if state.get('_observable'):
            _format['style'] = 'filled'
        dot.node(n, **_format)

    # add edges to graph
    for u, v, label in G.edges_iter(data='param', default=''):
        if not internal and (u[0] == '_' or v[0] == '_'):
            continue

        label = label if param_names else ''
        dot.edge(u, v, str(label))

    if filename is not None:
        try:
            filebase, filetype = filename.split('.')
            dot.format = filetype
            dot.render(filebase)
        except:
            raise ValueError('Saving to file {} failed.'.format(filename))

    return dot

