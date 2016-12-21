from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt

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


def _create_axes(axes, shape, **kwargs):
    """Checks the axes

    Parameters
    ----------
    axes : one or an iterable of plt.Axes
    shape : tuple of ints (x,) or (x,y)

    Returns
    -------
    axes : np.array of plt.Axes
    kwargs : dict
        Input kwargs without items related to creating a figure.
    """
    if axes is not None:
        axes = np.atleast_1d(axes)
        if axes.shape != shape:
            raise ValueError("Shape of axes does not match the given shape!")
    else:
        fig_kwargs = {}
        for k in ['figsize', 'sharex', 'sharey', 'dpi', 'num', 'facecolor', 'edgecolor']:
            if k in kwargs.keys():
                fig_kwargs[k] = kwargs.pop(k)
        fig, axes = plt.subplots(ncols=shape[1], nrows=shape[0], **fig_kwargs)
        axes = np.atleast_1d(axes)
    return axes, kwargs


def plot_histogram(samples, bins=20, axes=None, **kwargs):
    """

    Parameters
    ----------
    samples : dict of np.arrays
    bins : int, optional
    axes : one or an iterable of plt.Axes, optional

    Returns
    -------
    axes : np.array of plt.Axes
    """
    shape = (max(1, len(samples) // 4), 4)
    axes, kwargs = _create_axes(axes, shape, **kwargs)
    for ii, k in enumerate(samples.keys()):
        axes[ii].hist(samples[k], bins=bins, **kwargs)
        axes[ii].set_xlabel(k)
    return axes
