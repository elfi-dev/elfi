import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from elfi.model.elfi_model import ElfiModel, NodeReference, Constant
import elfi.visualization.interactive as visin


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


def plot_state_1d(model_bo):
    """Plot the GP surface and the acquisition function in 1D.

    Notes
    -----
    The method is experimental.

    Parameters
    ----------
    model_bo : elfi.methods.parameter_inference.BOLFI
    """

    # Defining plotting settings
    fig, arr_ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4),
        sharex=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
    fig.tight_layout(pad=2.0)

    gp = model_bo.target_model
    arr_x = np.linspace(*gp.bounds[0])

    # Plotting the GP's mean function.
    arr_ax[0].plot(arr_x, gp.predict_mean(arr_x))
    arr_ax[0].scatter(gp.X, gp.Y)
    arr_ax[0].set_title('GP target surface')
    arr_ax[0].set_xlabel(model_bo.parameter_names[0])
    arr_ax[0].set_ylabel('Discrepancy')

    # Plotting the acquisition function.
    fn_acq = lambda x: model_bo.acquisition_method.evaluate(arr_x, len(gp.X))
    arr_ax[1].plot(arr_x, fn_acq(arr_x))
    arr_ax[1].set_title('Acquisition surface')
    arr_ax[1].set_xlabel(model_bo.parameter_names[0])
    arr_ax[1].set_ylabel('Acquisition score')


def plot_state_2d(model_bo, **options):
    """Plot the GP surface and the acquisition function in 2D.

    Notes
    -----
    The method is experimental.

    Parameters
    ----------
    model_bo : elfi.methods.parameter_inference.BOLFI
    """
    fig, arr_ax = plt.subplots(1, 2, figsize=(13,6), sharex='row',
            sharey='row')

    gp = model_bo.target_model

    # Draw the GP surface
    visin.draw_contour(gp.predict_mean,
                       gp.bounds,
                       model_bo.parameter_names,
                       title='GP target surface',
                       points=gp.X,
                       axes=arr_ax[0], **options)
    # Draw the latest acquisitions
    if options.get('interactive'):
        point = gp.X[-1, :]
        arr_ax[1].scatter(*point, color='red')

    displays = [gp._gp]
    if options.get('interactive'):
        from IPython import display
        displays.insert(0, display.HTML(
                '<span><b>Iteration {}:</b> Acquired {} at {}</span>'.format(
                    len(gp.Y), gp.Y[-1][0], point)))
    # Update
    visin._update_interactive(displays, options)

    fn_acq = lambda x: model_bo.acquisition_method.evaluate(x, len(gp.X))
    # Draw the acquisition surface
    visin.draw_contour(fn_acq,
                       gp.bounds,
                       model_bo.parameter_names,
                       title='Acquisition surface',
                       points=None,
                       axes=arr_ax[1], **options)

    if options.get('close'):
        plt.close()


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
        fig.tight_layout(pad=2.0)
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


def plot_pairs(data, selector=None, bins=20, axes=None, **kwargs):
    """Plot pair-wise relationships in a grid with marginals on the diagonal.

    Notes
    -----
    Removed: The y-axis of marginal histograms are scaled.

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

    # Pop the target kwargs.
    edgecolor = kwargs.pop('edgecolor', 'none')
    dot_size = kwargs.pop('s', 25)

    # Filter the data.
    data_selected = _limit_params(data, selector)

    # Initialise the figure.
    shape_fig = (len(data_selected), len(data_selected))
    axes, kwargs = _create_axes(axes, shape_fig, **kwargs)

    # Populate the grid of figures.
    for idx_row, key_row in enumerate(data_selected):
        for idx_col, key_col in enumerate(data_selected):
            if idx_row == idx_col:
                # Plot the marginals.
                axes[idx_row, idx_col].hist(data_selected[key_row], bins=bins, **kwargs)
                axes[idx_row, idx_col].set_xlabel(key_row)
                # Experimental: Calculate the bin length.
                x_min, x_max = axes[idx_row, idx_col].get_xlim()
                length_bin = (x_max - x_min)/bins
                axes[idx_row, idx_col].set_ylabel(
                    'Count (bin length: {0:.2f})'.format(length_bin))
            else:
                # Plot the pairs.
                axes[idx_row, idx_col].scatter(data_selected[key_row],
                    data_selected[key_col], edgecolor=edgecolor, s=dot_size,
                    **kwargs)
                axes[idx_row, idx_col].set_xlabel(key_row)
                axes[idx_row, idx_col].set_ylabel(key_col)

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
