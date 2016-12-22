from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

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
    fig_kwargs = {}
    kwargs['figsize'] = kwargs.get('figsize', (16, 4*shape[0]))
    for k in ['figsize', 'sharex', 'sharey', 'dpi', 'num']:
        if k in kwargs.keys():
            fig_kwargs[k] = kwargs.pop(k)

    if axes is not None:
        axes = np.atleast_1d(axes)
    else:
        fig, axes = plt.subplots(ncols=shape[1], nrows=shape[0], **fig_kwargs)
        axes = np.atleast_1d(axes)
    return axes, kwargs


def _limit_params(samples, selector=None):
    """Pick selected samples.

    Parameters
    ----------
    samples : OrderedDict of np.arrays
    selector : iterable of ints or strings, optional
        Indices or keys to use from samples. Default to all.

    Returns
    -------
    selected : OrderedDict of np.arrays
    """
    if selector is None:
        return samples
    else:
        selected = OrderedDict()
        for ii, k in enumerate(samples):
            if ii in selector or k in selector:
                selected[k] = samples[k]
        return selected


def plot_marginals(samples, selector=None, bins=20, axes=None, **kwargs):
    """Plot marginal distributions for parameters.

    Parameters
    ----------
    samples : OrderedDict of np.arrays
    selector : iterable of ints or strings, optional
        Indices or keys to use from samples. Default to all.
    bins : int, optional
        Number of bins in histogram.
    axes : one or an iterable of plt.Axes, optional

    Returns
    -------
    axes : np.array of plt.Axes
    """
    samples = _limit_params(samples, selector)
    ncols = kwargs.pop('ncols', 5)
    nrows = kwargs.pop('nrows', 1)
    kwargs['sharey'] = kwargs.get('sharey', True)
    shape = (max(1, len(samples) // ncols), min(len(samples), ncols))
    axes, kwargs = _create_axes(axes, shape, **kwargs)
    axes = axes.ravel()
    for ii, k in enumerate(samples.keys()):
        axes[ii].hist(samples[k], bins=bins, **kwargs)
        axes[ii].set_xlabel(k)

    return axes


def plot_pairs(samples, selector=None, bins=20, axes=None, **kwargs):
    """Plot pairwise relationships as a matrix with marginals on the diagonal.

    The y-axis of marginal histograms are scaled.

     Parameters
    ----------
    samples : OrderedDict of np.arrays
    selector : iterable of ints or strings, optional
        Indices or keys to use from samples. Default to all.
    bins : int, optional
        Number of bins in histograms.
    axes : one or an iterable of plt.Axes, optional

    Returns
    -------
    axes : np.array of plt.Axes
    """
    samples = _limit_params(samples, selector)
    shape = (len(samples), len(samples))
    edgecolor = kwargs.pop('edgecolor', 'none')
    dot_size = kwargs.pop('s', 2)
    kwargs['sharex'] = kwargs.get('sharex', 'col')
    kwargs['sharey'] = kwargs.get('sharey', 'row')
    axes, kwargs = _create_axes(axes, shape, **kwargs)

    for i1, k1 in enumerate(samples):
        min_samples = samples[k1].min()
        max_samples = samples[k1].max()
        for i2, k2 in enumerate(samples):
            if i1 == i2:
                # create a histogram with scaled y-axis
                hist, bin_edges = np.histogram(samples[k1], bins=bins)
                bar_width = bin_edges[1] - bin_edges[0]
                hist = (hist - hist.min()) * (max_samples - min_samples) / (hist.max() - hist.min())
                axes[i1, i2].bar(bin_edges[:-1], hist, bar_width, bottom=min_samples, **kwargs)
            else:
                axes[i1, i2].scatter(samples[k2], samples[k1], s=dot_size, edgecolor=edgecolor, **kwargs)

        axes[i1, 0].set_ylabel(k1)
        axes[-1, i1].set_xlabel(k1)

    return axes
