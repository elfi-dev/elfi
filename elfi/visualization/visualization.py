import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from elfi.model.elfi_model import ElfiModel, NodeReference, Constant
from elfi import utils


def nx_draw(G, internal=False, param_names=False, filename=None, format=None):
    """
    Draw the `ElfiModel`.

    Parameters
    ----------
    G : nx.DiGraph or ElfiModel
        Graph or model to draw
    internal : boolean, optional
        Whether to draw internal nodes (starting with an underscore)
    param_names : bool, optional
        Show param names on edges
    filename : str, optional
        If given, save the dot file into the given filename.
    format : str, optional
        format of the file

    Notes
    -----
    Requires the optional 'graphviz' library.

    Returns
    -------
    dot
        A GraphViz dot representation of the model.

    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("The graphviz library is required for this feature.")

    if isinstance(G, ElfiModel):
        G = G.source_net
    elif isinstance(G, NodeReference):
        G = G.model.source_net

    dot = Digraph(format=format)

    hidden = set()

    for n, state in G.nodes_iter(data=True):
        if not internal and n[0] == '_' and state.get('_class') == Constant:
            hidden.add(n)
            continue
        _format = {'shape': 'circle', 'fillcolor': 'gray80', 'style': 'solid'}
        if state.get('_observable'):
            _format['style'] = 'filled'
        dot.node(n, **_format)

    # add edges to graph
    for u, v, label in G.edges_iter(data='param', default=''):
        if not internal and u in hidden:
            continue

        label = label if param_names else ''
        dot.edge(u, v, str(label))

    if filename is not None:
        dot.render(filename)

    return dot


def _create_axes(axes, shape, **kwargs):
    """Checks the axes and creates them if necessary.

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
    """Pick only the selected parameters from all samples.

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


def plot_traces(result, selector=None, axes=None, **kwargs):
    """Trace plot for MCMC samples.

    The black vertical lines indicate the used warmup.

    Parameters
    ----------
    result : Result_BOLFI
    selector : iterable of ints or strings, optional
        Indices or keys to use from samples. Default to all.
    axes : one or an iterable of plt.Axes, optional
    kwargs

    Returns
    -------
    axes : np.array of plt.Axes
    """
    samples_sel = _limit_params(result.samples, selector)
    shape = (len(samples_sel), result.n_chains)
    kwargs['sharex'] = 'all'
    kwargs['sharey'] = 'row'
    axes, kwargs = _create_axes(axes, shape, **kwargs)

    i1 = 0
    for i2, k in enumerate(result.samples):
        if k in samples_sel:
            for i3 in range(result.n_chains):
                axes[i1, i3].plot(result.chains[i3, :, i2], **kwargs)
                axes[i1, i3].axvline(result.warmup, color='black')

            axes[i1, 0].set_ylabel(k)
            i1 += 1

    for ii in range(result.n_chains):
        axes[-1, ii].set_xlabel('Iterations in Chain {}'.format(ii))

    return axes


def _diff(x, y):
    return x - y


def _abs_diff(x, y):
    return abs(_diff(x, y))


def compare(funs, *args, comparison='absdiff'):
    """Plot a comparison of two functions.

    Parameters
    ----------
    funs : list
      The functions to plot.
    *args
      The points along each axis to plot.
    comparison : str or function
      How to compare the functions? Can be a 'absdiff' for absolute differences
      or 'diff' for differences. You can also pass a custom function to perform
      the comparison. The default is 'absdiff'.

    Examples
    --------
    Plot the absolute difference of sine and cosine.
    >>> compare([np.sin, np.cos], np.linspace(-3, 3, 100))
    """
    str_comp = {'absdiff': _abs_diff,
                'diff': _diff}

    if isinstance(comparison, str):
        comp = str_comp.get(comparison, None) or _abs_diff
    elif callable(comparison):
        comp = comparison

    grid, res = utils.tabulate(funs, *args)
    if len(args) == 1:
        plt.plot(grid, comp(*res))
    elif len(args) == 2:
        plt.contourf(*grid, comp(*res))
    else:
        raise ValueError("Cannot plot in {} dimensions.".format(len(args)))


def overlay(funs, *args):
    """Overlay plots of functions.

    Parameters
    ----------
    funs : list or dict
      The functions to plot. The plotting options for each function
      can be specified by passing a dictionary with the functions as the keys.
    *args
      The points along each axis to plot.

    Examples
    --------
    Plot with default settings:
    >>> overlay([np.sin, np.cos], np.linspace(-3, 3, 100))

    Plot with custom settings:
    >>> overlay({np.sin:{}, np.cos:{'linestyle': 'dashed'}}, np.linspace(-3, 3, 100))

    Plot a contour plot:
    >>> import scipy.stats as ss
    >>> overlay([ss.multivariate_normal(mean=np.zeros(2)).pdf], np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    """
    if isinstance(funs, (list, tuple)):
        funs = dict(zip(funs, [{} for f in funs]))

    grid, res = utils.tabulate(list(funs.keys()), *args)
    fig, ax = plt.subplots()

    if len(args) == 1:
        for i, f in enumerate(funs):
            ax.plot(grid, res[i], **funs.get(f, None))
    elif len(args) == 2:
        for i, f in enumerate(funs):
            ax.contour(*grid, res[i], **funs.get(f, None))
    else:
        raise ValueError("Cannot plot in {} dimensions.".format(len(args)))
