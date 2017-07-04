"""This module contains functions for interactive ("iterative") plotting."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from IPython import display

logger = logging.getLogger(__name__)


def plot_sample(samples, nodes=None, n=-1, displays=None, **opts):
    """
    Experimental, only dims 1-2 supported.

    Parameters
    ----------
    samples : Sample
    nodes : str or list[str], optional
    n : int, optional
        Number of plotted samples [0, n).
    displays : IPython.display.HTML

    """
    axes = _prepare_axes(opts)
    if samples is None:
        return
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

    if opts.get('interactive'):
        update_interactive(displays, opts)
        plt.close()


def get_axes(**opts):
    """Get an Axes object from `options`, or create one if needed."""
    if 'axes' in opts:
        return opts['axes']
    return plt.gca()


def update_interactive(displays, opts):
    """Update the interactive plot."""
    displays = displays or []
    if opts.get('interactive'):
        display.clear_output(wait=True)
        displays.insert(0, plt.gcf())
        display.display(*displays)


def _prepare_axes(opts):
    axes = get_axes(**opts)
    ion = opts.get('interactive')

    if ion:
        axes.clear()
    if opts.get('xlim'):
        axes.set_xlim(opts.get('xlim'))
    if opts.get('ylim'):
        axes.set_ylim(opts.get('ylim'))

    return axes


def draw_contour(fn, bounds, params=None, points=None, title=None, **opts):
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
    # Preparing the contour plot settings.
    if opts.get('axes'):
        axes = opts['axes']
        plt.sca(axes)
    x, y = np.meshgrid(np.linspace(*bounds[0]), np.linspace(*bounds[1]))
    z = fn(np.c_[x.reshape(-1), y.reshape(-1)])

    # Plotting the contour.
    obj_contour = plt.contour(x, y, z.reshape(x.shape))
    plt.clabel(obj_contour, inline=1, fontsize=12, fmt='%.0e', colors='w')
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(x, y)
    plt.imshow(zi,
               vmin=z.min(),
               vmax=z.max(),
               origin='lower',
               extent=[x.min(), x.max(), y.min(), y.max()])

    # Adding the acquisition points.
    if points is not None:
        plt.scatter(points[:, 0], points[:, 1], color='k')

    # Adding the labels.
    if title:
        plt.title(title)
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    if params:
        plt.xlabel(params[0])
        plt.ylabel(params[1])
