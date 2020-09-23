"""This module includes common functions for visualization."""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

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

    for n, state in G.nodes(data=True):
        if not internal and n[0] == '_' and state['attr_dict'].get('_class') == Constant:
            hidden.add(n)
            continue
        _format = {'shape': 'circle', 'fillcolor': 'gray80', 'style': 'solid'}
        if state.get('_observable'):
            _format['style'] = 'filled'
        dot.node(n, **_format)

    # add edges to graph
    for u, v, label in G.edges(data='param', default=''):
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
        axes = np.atleast_2d(axes)
    else:
        fig, axes = plt.subplots(ncols=shape[1], nrows=shape[0], **fig_kwargs)
        axes = np.atleast_2d(axes)
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
    ncols = len(samples.keys()) if len(samples.keys()) > 5 else 5
    ncols = kwargs.pop('ncols', ncols)
    samples = _limit_params(samples, selector)
    shape = (max(1, len(samples) // ncols), min(len(samples), ncols))
    axes, kwargs = _create_axes(axes, shape, **kwargs)
    axes = axes.ravel()
    for idx, key in enumerate(samples.keys()):
        axes[idx].hist(samples[key], bins=bins, **kwargs)
        axes[idx].set_xlabel(key)

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
    axes, kwargs = _create_axes(axes, shape, **kwargs)

    for idx_row, key_row in enumerate(samples):
        min_samples = samples[key_row].min()
        max_samples = samples[key_row].max()
        for idx_col, key_col in enumerate(samples):
            if idx_row == idx_col:
                # create a histogram with scaled y-axis
                hist, bin_edges = np.histogram(samples[key_row], bins=bins)
                bar_width = bin_edges[1] - bin_edges[0]
                hist = (hist - hist.min()) * (max_samples - min_samples) / (
                    hist.max() - hist.min())
                axes[idx_row, idx_col].bar(bin_edges[:-1],
                                           hist,
                                           bar_width,
                                           bottom=min_samples,
                                           **kwargs)
            else:
                axes[idx_row, idx_col].scatter(samples[key_col],
                                               samples[key_row],
                                               s=dot_size,
                                               edgecolor=edgecolor,
                                               **kwargs)

        axes[idx_row, 0].set_ylabel(key_row)
        axes[-1, idx_row].set_xlabel(key_row)

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


class ProgressBar:
    """Progress bar monitoring the inference process.

    Attributes
    ----------
    prefix : str, optional
        Prefix string
    suffix : str, optional
        Suffix string
    decimals : int, optional
        Positive number of decimals in percent complete
    length : int, optional
        Character length of bar
    fill : str, optional
        Bar fill character
    scaling : int, optional
        Integer used to scale current iteration and total iterations of the progress bar

    """

    def __init__(self, prefix='', suffix='', decimals=1, length=100, fill='='):
        """Construct progressbar for monitoring.

        Parameters
        ----------
        prefix : str, optional
            Prefix string
        suffix : str, optional
            Suffix string
        decimals : int, optional
            Positive number of decimals in percent complete
        length : int, optional
            Character length of bar
        fill : str, optional
            Bar fill character

        """
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = 1
        self.length = length
        self.fill = fill
        self.scaling = 0

    def update_progressbar(self, iteration, total):
        """Print updated progress bar in console.

        Parameters
        ----------
        iteration : int
            Integer indicating completed iterations
        total : int
            Integer indicating total number of iterations

        """
        if total - self.scaling > 0:
            percent = ("{0:." + str(self.decimals) + "f}").\
                format(100 * ((iteration - self.scaling) / float(total - self.scaling)))
            filled_length = int(self.length * (iteration - self.scaling) // (total - self.scaling))
            bar = self.fill * filled_length + '-' * (self.length - filled_length)
            print('%s [%s] %s%% %s' % (self.prefix, bar, percent, self.suffix), end='\r')
            if iteration == total:
                print()

    def reinit_progressbar(self, scaling=0, reinit_msg=None):
        """Reinitialize new round of progress bar.

        Parameters
        ----------
        scaling : int, optional
            Integer used to scale current and total iterations of the progress bar
        reinit_msg : str, optional
            Message printed before restarting an empty progess bar on a new line

        """
        self.scaling = scaling
        if scaling == 0:
            print(reinit_msg)
        else:
            self.update_progressbar(scaling + 1, scaling + 1)
            print('\n' + reinit_msg)


def plot_params_vs_node(node, n_samples=100, func=None, seed=None, axes=None, **kwargs):
    """Plot some realizations of parameters vs. `node`.

    Useful e.g. for exploring how a summary statistic varies with parameters.
    Currently only nodes with scalar output are supported, though a function `func` can
    be given to reduce node output. This allows giving the simulator as the `node` and
    applying a summarizing function without incorporating it into the ELFI graph.

    If `node` is one of the model parameters, its histogram is plotted.

    Parameters
    ----------
    node : elfi.NodeReference
        The node which to evaluate. Its output must be scalar (shape=(batch_size,1)).
    n_samples : int, optional
        How many samples to plot.
    func : callable, optional
        A function to apply to node output.
    seed : int, optional
    axes : one or an iterable of plt.Axes, optional

    Returns
    -------
    axes : np.array of plt.Axes

    """
    model = node.model
    parameters = model.parameter_names
    node_name = node.name

    if node_name in parameters:
        outputs = [node_name]
        shape = (1, 1)
        bins = kwargs.pop('bins', 20)

    else:
        outputs = parameters + [node_name]
        n_params = len(parameters)
        ncols = n_params if n_params < 5 else 5
        ncols = kwargs.pop('ncols', ncols)
        edgecolor = kwargs.pop('edgecolor', 'none')
        dot_size = kwargs.pop('s', 20)
        shape = (1 + n_params // (ncols + 1), ncols)

    data = model.generate(batch_size=n_samples, outputs=outputs, seed=seed)

    if func is not None:
        if hasattr(func, '__name__'):
            node_name = func.__name__
        else:
            node_name = 'func'
        data[node_name] = func(data[node.name])  # leaves rest of the code unmodified

    if data[node_name].shape != (n_samples,):
        raise NotImplementedError("The plotted quantity must have shape ({},), was {}."
                                  .format(n_samples, data[node_name].shape))

    axes, kwargs = _create_axes(axes, shape, sharey=True, **kwargs)
    axes = axes.ravel()

    if len(outputs) == 1:
        axes[0].hist(data[node_name], bins=bins, normed=True)
        axes[0].set_xlabel(node_name)

    else:
        for idx, key in enumerate(parameters):
            axes[idx].scatter(data[key],
                              data[node_name],
                              s=dot_size,
                              edgecolor=edgecolor,
                              **kwargs)

            axes[idx].set_xlabel(key)
        axes[0].set_ylabel(node_name)

        for idx in range(len(parameters), len(axes)):
            axes[idx].set_axis_off()

    return axes


def plot_discrepancy(gp, parameter_names, axes=None, **kwargs):
    """Plot acquired parameters vs. resulting discrepancy.

    Parameters
    ----------
    axes : plt.Axes or arraylike of plt.Axes
    gp : GPyRegression target model, required
    parameter_names : dict, required
        Parameter names from model.parameters dict('parameter_name':(lower, upper), ... )`

    Returns
    -------
    axes : np.array of plt.Axes

    """
    n_plots = gp.input_dim
    ncols = len(gp.bounds) if len(gp.bounds) < 5 else 5
    ncols = kwargs.pop('ncols', ncols)
    kwargs['sharey'] = kwargs.get('sharey', True)
    if n_plots > 10:
        shape = (1 + (1 + n_plots) // (ncols + 1), ncols)
    else:
        shape = (1 + n_plots // (ncols + 1), ncols)
    axes, kwargs = _create_axes(axes, shape, **kwargs)
    axes = axes.ravel()

    for ii in range(n_plots):
        axes[ii].scatter(gp.X[:, ii], gp.Y[:, 0], **kwargs)
        axes[ii].set_xlabel(parameter_names[ii])
        if ii % ncols == 0:
            axes[ii].set_ylabel('Discrepancy')

    for idx in range(len(parameter_names), len(axes)):
        axes[idx].set_axis_off()

    return axes


def plot_gp(gp, parameter_names, axes=None, resol=50,
            const=None, bounds=None, true_params=None, **kwargs):
    """Plot pairwise relationships as a matrix with parameters vs. discrepancy.

    Parameters
    ----------
    gp : GPyRegression, required
    parameter_names : list, required
        Parameter names in format ['mu_0', 'mu_1', ..]
    axes : plt.Axes or arraylike of plt.Axes
    resol : int, optional
        Resolution of the plotted grid.
    const : np.array, optional
        Values for parameters in plots where held constant. Defaults to minimum evidence.
    bounds: list of tuples, optional
        List of tuples for axis boundaries.
    true_params : dict, optional
        Dictionary containing parameter names with corresponding true parameter values.

    Returns
    -------
    axes : np.array of plt.Axes

    """
    n_plots = gp.input_dim
    shape = (n_plots, n_plots)
    axes, kwargs = _create_axes(axes, shape, **kwargs)

    x_evidence = gp.X
    y_evidence = gp.Y
    if const is None:
        const = x_evidence[np.argmin(y_evidence), :]
    bounds = bounds or gp.bounds

    cmap = plt.cm.get_cmap("bone")

    plt.subplots_adjust(wspace=0.2, hspace=0.0, left=0.3, right=0.7, top=0.8, bottom=0.05)
    for ix in range(n_plots):
        for jy in range(n_plots):
            if ix == jy:
                axes[jy, ix].scatter(x_evidence[:, ix], y_evidence)
                axes[jy, ix].set_aspect(aspect=(bounds[ix][1] - bounds[ix][0]) /
                                               (max(y_evidence) - min(y_evidence)))
                axes[jy, ix].get_yaxis().set_ticklabels([])
                axes[jy, ix].yaxis.tick_right()
                axes[jy, ix].set_ylabel('Discrepancy')
                axes[jy, ix].yaxis.set_label_position("right")

                if true_params is not None:
                    axes[jy, ix].plot([true_params[parameter_names[ix]],
                                      true_params[parameter_names[ix]]],
                                      [min(y_evidence), max(y_evidence)],
                                      color='orange', alpha=0.5, linewidth=4)
                axes[jy, ix].axis([bounds[ix][0], bounds[ix][1], min(y_evidence), max(y_evidence)])

            elif ix < jy:
                x1 = np.linspace(bounds[ix][0], bounds[ix][1], resol)
                y1 = np.linspace(bounds[jy][0], bounds[jy][1], resol)
                x, y = np.meshgrid(x1, y1)
                predictors = np.tile(const, (resol * resol, 1))
                predictors[:, ix] = x.ravel()
                predictors[:, jy] = y.ravel()

                z = gp.predict_mean(predictors).reshape(resol, resol)
                axes[jy, ix].contourf(x, y, z, cmap=cmap)
                axes[jy, ix].scatter(x_evidence[:, ix], x_evidence[:, jy], color="red", alpha=0.1)
                axes[jy, ix].set_aspect(aspect=(bounds[ix][1] - bounds[ix][0]) /
                                               (bounds[jy][1] - bounds[jy][0]))

                if true_params is not None:
                    axes[jy, ix].plot([true_params[parameter_names[ix]],
                                      true_params[parameter_names[ix]]],
                                      [bounds[jy][0], bounds[jy][1]],
                                      color='orange', alpha=0.5, linewidth=4)

                    axes[jy, ix].plot([bounds[ix][0], bounds[ix][1]],
                                      [true_params[parameter_names[jy]],
                                      true_params[parameter_names[jy]]],
                                      color='orange', alpha=0.5, linewidth=4)

                if ix == 0:
                    axes[jy, ix].set_ylabel(parameter_names[jy])
                else:
                    axes[jy, ix].get_yaxis().set_ticklabels([])

                axes[jy, ix].axis([bounds[ix][0], bounds[ix][1], bounds[jy][0], bounds[jy][1]])

            else:
                axes[jy, ix].axis('off')

            if jy < n_plots-1:
                axes[jy, ix].get_xaxis().set_ticklabels([])
            else:
                axes[jy, ix].set_xlabel(parameter_names[ix])

    return axes
