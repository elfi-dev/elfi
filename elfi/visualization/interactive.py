"""This module contains functions for interactive ("iterative") plotting."""

import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_sample(samples, nodes=None, n=-1, displays=None, **options):
    """Plot a scatterplot of samples.

    Experimental, only dims 1-2 supported.

    Parameters
    ----------
    samples : Sample
    nodes : str or list[str], optional
    n : int, optional
        Number of plotted samples [0, n).
    displays : IPython.display.HTML

    """
    axes = _prepare_axes(options)

    nodes = nodes or sorted(samples.keys())[:2]
    if isinstance(nodes, str):
        nodes = [nodes]

    if len(nodes) == 1:
        axes.set_xlabel(nodes[0])
        axes.hist(samples[nodes[0]][:n])
    else:
        if len(nodes) > 2:
            logger.warning('Over 2-dimensional plots not supported. Falling back to 2d'
                           'projection.')
        axes.set_xlabel(nodes[0])
        axes.set_ylabel(nodes[1])
        axes.scatter(samples[nodes[0]][:n], samples[nodes[1]][:n])

    _update_interactive(displays, options)

    if options.get('close'):
        plt.close()


def get_axes(**options):
    """Get an Axes object from `options`, or create one if needed."""
    if 'axes' in options:
        return options['axes']
    return plt.gca()


def _update_interactive(displays, options):
    displays = displays or []
    if options.get('interactive'):
        from IPython import display
        display.clear_output(wait=True)
        displays.insert(0, plt.gcf())
        display.display(*displays)


def _prepare_axes(options):
    axes = get_axes(**options)
    ion = options.get('interactive')

    if ion:
        axes.clear()

    if options.get('xlim'):
        axes.set_xlim(options.get('xlim'))
    if options.get('ylim'):
        axes.set_ylim(options.get('ylim'))

    return axes


def draw_contour(fn, bounds, nodes=None, points=None, title=None, **options):
    """Plot a contour of a function.

    Experimental, only 2D supported.

    Parameters
    ----------
    fn : callable
    bounds : list[arraylike]
        Bounds for the plot, e.g. [(0, 1), (0,1)].
    nodes : list[str], optional
    points : arraylike, optional
        Additional points to plot.
    title : str, optional

    """
    ax = get_axes(**options)

    x, y = np.meshgrid(np.linspace(*bounds[0]), np.linspace(*bounds[1]))
    z = fn(np.c_[x.reshape(-1), y.reshape(-1)])

    if ax:
        plt.sca(ax)
    plt.cla()

    if title:
        plt.title(title)
    try:
        plt.contour(x, y, z.reshape(x.shape))
    except ValueError:
        logger.warning('Could not draw a contour plot')
    if points is not None:
        plt.scatter(points[:-1, 0], points[:-1, 1])
        if options.get('interactive'):
            plt.scatter(points[-1, 0], points[-1, 1], color='r')

    plt.xlim(bounds[0])
    plt.ylim(bounds[1])

    if nodes:
        plt.xlabel(nodes[0])
        plt.ylabel(nodes[1])
