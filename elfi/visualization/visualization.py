"""This module includes common functions for visualization."""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import elfi.visualization.interactive as visin
from elfi.model.elfi_model import Constant, ElfiModel, NodeReference


def nx_draw(G, internal=False, param_names=False, filename=None, format=None):
    """Draw the `ElfiModel`.

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
    """Check the axes and create them if necessary.

    Parameters
    ----------
    axes : plt.Axes or arraylike of plt.Axes
    shape : tuple of int
        (x,) or (x,y)
    kwargs

    Returns
    -------
    axes : np.array of plt.Axes
    kwargs : dict
        Input kwargs without items related to creating a figure.

    """
    fig_kwargs = {}
    kwargs['figsize'] = kwargs.get('figsize', (16, 4 * shape[0]))
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
    kwargs['sharey'] = kwargs.get('sharey', True)
    shape = (max(1, len(samples) // ncols), min(len(samples), ncols))
    axes, kwargs = _create_axes(axes, shape, **kwargs)
    axes = axes.ravel()
    for ii, k in enumerate(samples.keys()):
        axes[ii].hist(samples[k], bins=bins, **kwargs)
        axes[ii].set_xlabel(k)

    return axes


def plot_state_1d(model_bo, arr_ax=None, **options):
    """Plot the GP surface and the acquisition function in 1-D.

    Notes
    -----
    The method is experimental.

    Parameters
    ----------
    model_bo : elfi.methods.parameter_inference.BOLFI
    arr_ax : array_like, optional

    Returns
    -------
    array_like
        Axes for interactive visualisation

    """
    gp = model_bo.target_model
    pts_eval = np.linspace(*gp.bounds[0])

    if arr_ax is None:
        fig, arr_ax = plt.subplots(nrows=1,
                                   ncols=2,
                                   figsize=(12, 4),
                                   sharex=True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(-3, 4))
        fig.tight_layout(pad=2.0)

        # Plotting the acquisition space.
        arr_ax[1].set_title('Acquisition surface')
        arr_ax[1].set_xlabel(model_bo.parameter_names[0])
        arr_ax[1].set_ylabel(options.pop('method_acq'))
        score_acq = model_bo.acquisition_method.evaluate(pts_eval)
        arr_ax[1].legend(loc='upper right')
        arr_ax[1].plot(pts_eval,
                       score_acq,
                       color='k',
                       label='acquisition function')

        # Plotting the confidence interval and the mean.
        mean, var = gp.predict(pts_eval, noiseless=False)
        sigma = np.sqrt(var)
        z_95 = 1.96
        lb_ci = mean - z_95 * (sigma)
        ub_ci = mean + z_95 * (sigma)
        arr_ax[0].fill(np.concatenate([pts_eval, pts_eval[::-1]]),
                       np.concatenate([lb_ci, ub_ci[::-1]]),
                       alpha=.1,
                       fc='k',
                       ec='None',
                       label='95% confidence interval')
        arr_ax[0].plot(pts_eval, mean, color='k', label='mean')

        # Plotting the acquisition threshold, epsilon.
        if model_bo.acquisition_method.name in ['max_var', 'rand_max_var', 'exp_int_var']:
            thresh_acq = np.repeat(model_bo.acquisition_method.eps,
                                   len(pts_eval))
            arr_ax[0].plot(pts_eval,
                           thresh_acq,
                           color='g',
                           label='acquisition threshold')

        # Plotting the acquired points.
        arr_ax[0].set_title('GP target surface')
        arr_ax[0].set_xlabel(model_bo.parameter_names[0])
        arr_ax[0].set_ylabel('Discrepancy')
        arr_ax[0].legend(loc='upper right')
        arr_ax[0].scatter(gp.X, gp.Y, color='k')

        return arr_ax
    else:
        if options.get('interactive'):
            from IPython import display
            pt_last = options.pop('point_acq')

            # Plotting the last acquired point on the GP target surface.
            arr_ax[0].scatter(pt_last['x'], pt_last['d'], color='r')

            # Plotting the lines indicating the acquisition's location on the acquisition space.
            ymin, ymax = arr_ax[1].get_ylim()
            arr_ax[1].vlines(x=pt_last['x'], ymin=ymin, ymax=ymax,
                             color='r', linestyle='--',
                             label='latest acquisition')

            # Handling the interactive display.
            displays = []
            displays.append(gp.instance)
            n_it = int(len(gp.Y) / model_bo.batch_size)
            html_disp = '<span><b>Iteration {}:</b> Acquired {} at {}</span>' \
                .format(n_it, pt_last['d'], pt_last['x'])
            displays.append(display.HTML(html_disp))
            visin.update_interactive(displays, options=options)

            plt.close()


def plot_state_2d(model_bo, arr_ax=None, **options):
    """Plot the GP surface and the acquisition function in 2-D.

    Notes
    -----
    The method is experimental.

    Parameters
    ----------
    model_bo : elfi.methods.parameter_inference.BOLFI
    arr_ax : array_like, optional

    Returns
    -------
    array_like
        Axes for interactive visualisation

    """
    gp = model_bo.target_model

    if arr_ax is None:
        _, arr_ax = plt.subplots(nrows=1,
                                 ncols=2,
                                 figsize=(16, 10),
                                 sharex='row',
                                 sharey='row')

        # Plotting the acquisition space.
        def fn_acq(x):
            return model_bo.acquisition_method.evaluate(x, len(gp.X))
        visin.draw_contour(fn_acq,
                           gp.bounds,
                           model_bo.parameter_names,
                           title='Acquisition surface',
                           axes=arr_ax[1],
                           label=options.pop('method_acq'),
                           **options)

        # Plotting the GP target surface and the acquired points.
        visin.draw_contour(gp.predict_mean,
                           gp.bounds,
                           model_bo.parameter_names,
                           title='GP target surface',
                           points=gp.X,
                           axes=arr_ax[0],
                           label='Discrepancy',
                           **options)

        return arr_ax
    else:
        if options.get('interactive'):
            from IPython import display
            pt_last = options.pop('point_acq')

            # Plotting the last acquired point on the GP target surface and the acquisition space.
            arr_ax[0].scatter(pt_last['x'][:, 0], pt_last['x'][:, 1], color='r')
            arr_ax[1].scatter(pt_last['x'][:, 0], pt_last['x'][:, 1], color='r')

            # Handling the interactive display.
            displays = []
            displays.append(gp.instance)
            n_it = int(len(gp.Y) / model_bo.batch_size)
            html_disp = '<span><b>Iteration {}:</b> Acquired {} at {}</span>' \
                .format(n_it, pt_last['d'], pt_last['x'])
            displays.append(display.HTML(html_disp))
            visin.update_interactive(displays, options=options)

            plt.close()


def plot_pairs(data, selector=None, bins=20, axes=None, **kwargs):
    """Plot the pair-wise relationships in a grid with marginals on the diagonal.

    Notes
    -----
    Removed: The y-axis of marginal histograms are scaled.

    Parameters
    ----------
    data : OrderedDict of np.arrays
    selector : iterable of ints or strings, optional
        Indices or keys to use from samples. Default to all.
    bins : int, optional
        Number of bins in histograms.
    axes : one or an iterable of plt.Axes, optional

    Returns
    -------
    axes : np.array of plt.Axes

    """
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
                length_bin = (x_max - x_min) / bins
                axes[idx_row, idx_col].set_ylabel(
                    'Count (bin length: {0:.2f})'.format(length_bin))
            else:
                # Plot the pairs.
                axes[idx_row, idx_col].scatter(data_selected[key_row],
                                               data_selected[key_col],
                                               edgecolor=edgecolor,
                                               s=dot_size,
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
