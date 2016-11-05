import numpy as np
from graphviz import Digraph
import matplotlib.pyplot as plt

import elfi.core as core


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

    dot = Digraph()

    # add nodes to graph
    for n in nodes:
        node_format = {'shape': 'circle', 'fillcolor': 'grey', 'style': 'solid'}

        if hasattr(n, 'observed'):
            node_format['style'] = 'filled'

        dot.node(n.name, **node_format)

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


def surface_eval(fun, bounds, x_points=200, y_points=200):
    """Evaluate a function on a grid of points.

    Arguments
    ---------
    fun: (n, 2) -> float
        a function that maps an array of points in R^2 to R
    bounds: tuple
        box constraint for the region. For example ((0, 1), (0, 1))
    x_points: int
        number of points on the first axis
    y_points: int
        number of points on the second axis

    Returns
    -------
    A tuple (X, Y, Z) where X, Y are meshgrid components and Z is an array of
    function evaluations with shape (x_pints, y_points).
    """
    x = np.linspace(*bounds[0], x_points)
    y = np.linspace(*bounds[1], y_points)
    xx, yy = np.meshgrid(x, y)
    coords = np.array((xx.ravel(), yy.ravel())).T
    return xx, yy, fun(coords).reshape(len(x), len(y))


def contour_plot(fun, bounds, x_points=200, y_points=200, **kwargs):
    """Draw a contour plot of a function.

    Arguments
    ---------
    fun: (n, 2) -> float
        a function that maps an array of points in R^2 to R
    bounds: tuple
        box constraint for the region. For example ((0, 1), (0, 1))
    x_points: int
        number of points on the first axis
    y_points: int
        number of points on the second axis
    """
    del kwargs['dim']
    del kwargs['type']
    return plt.contour(*surface_eval(fun, bounds, x_points, y_points), **kwargs)


def surface_plot(fun, bounds, x_points=200, y_points=200, **kwargs):
    """Draw a surface plot of a function.

    Arguments
    ---------
    fun: (n, 2) -> float
        a function that maps an array of points in R^2 to R
    bounds: tuple
        box constraint for the region. For example ((0, 1), (0, 1))
    x_points: int
        number of points on the first axis
    y_points: int
        number of points on the second axis
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    del kwargs['dim']
    del kwargs['type']
    surf = ax.plot_surface(*surface_eval(fun, bounds, x_points, y_points), **kwargs)
    return fig


def plot_1d(fun, bounds, points=200, **kwargs):
    t = np.linspace(*bounds, points)
    return plt.plot(t, fun(t), **kwargs)


def plot(fun, bounds, type, **kwargs):
    dim = len(bounds)
    switch = {(1, 'surface'): plot_1d, (1, 'contour'): plot_1d,
              (2, 'surface'): surface_plot, (2, 'contour'): contour_plot}
    return switch.get((dim, type), plotting_error)(fun, bounds, dim=dim,
                                                   type=type, **kwargs)


def plotting_error(dim, type, **kwargs):
    raise ValueError("There is no {}-dimensional {} plot.".format(dim, type))
