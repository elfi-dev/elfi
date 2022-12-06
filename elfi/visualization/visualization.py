"""This module includes common functions for visualization."""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

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
        if state['attr_dict'].get('_observable'):
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
    kwargs['figsize'] = kwargs.get('figsize', (4 * shape[1], 4 * shape[0]))
    for k in ['figsize', 'sharex', 'sharey', 'dpi', 'num']:
        if k in kwargs.keys():
            fig_kwargs[k] = kwargs.pop(k)

    if axes is not None:
        axes = np.atleast_2d(axes)
    else:
        fig, axes = plt.subplots(ncols=shape[1], nrows=shape[0], **fig_kwargs)
        axes = np.reshape(axes, shape)
        fig.tight_layout(pad=2.0, h_pad=1.08, w_pad=1.08)
        fig.subplots_adjust(wspace=0.2, hspace=0.2)

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


def plot_marginals(samples, selector=None, bins=20, axes=None,
                   reference_value=None, **kwargs):
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
    ncols = len(samples.keys()) if len(samples.keys()) < 5 else 5
    ncols = kwargs.pop('ncols', ncols)
    samples = _limit_params(samples, selector)
    shape = (-(len(samples) // -ncols), min(len(samples), ncols))
    axes, kwargs = _create_axes(axes, shape, **kwargs)

    axes = axes.ravel()
    for idx, key in enumerate(samples.keys()):
        if reference_value is not None:
            axes[idx].plot(reference_value[key], 0,
                           color='red',
                           alpha=1.0,
                           linewidth=2,
                           marker='X',
                           clip_on=False,
                           markersize=12)
        if ('kde' in kwargs):
            kde = ss.gaussian_kde(samples[key])
            xs = np.linspace(min(samples[key]), max(samples[key]))
            axes[idx].plot(xs, kde(xs))
        else:
            axes[idx].hist(samples[key], bins=bins, **kwargs)
        axes[idx].set_xlabel(key)
    for idx in range(len(samples), len(axes)):
        axes[idx].set_axis_off()
    return axes


def plot_pairs(samples,
               selector=None,
               bins=20,
               reference_value=None,
               axes=None,
               draw_upper_triagonal=False,
               **kwargs):
    """Plot pairwise relationships as a matrix with marginals on the diagonal.

    The y-axis of marginal histograms are scaled.

    Parameters
    ----------
    samples : OrderedDict of np.arrays
    selector : iterable of ints or strings, optional
        Indices or keys to use from samples. Default to all.
    bins : int, optional
        Number of bins in histograms.
    reference_value: dict, optional
        Dictionary containing reference values for parameters.
    axes : one or an iterable of plt.Axes, optional
    draw_upper_triagonal: boolean, optional
        Boolean indicating whether to draw symmetric upper triagonal part.

    Returns
    -------
    axes : np.array of plt.Axes

    """
    samples = _limit_params(samples, selector)
    shape = (len(samples), len(samples))
    edgecolor = kwargs.pop('edgecolor', 'black')
    dot_size = kwargs.pop('s', 2)
    axes, kwargs = _create_axes(axes, shape, **kwargs)

    for idx_row, key_row in enumerate(samples):
        min_samples = samples[key_row].min()
        max_samples = samples[key_row].max()
        for idx_col, key_col in enumerate(samples):
            if idx_row == idx_col:
                axes[idx_row, idx_col].hist(samples[key_row], bins=bins, density=True, **kwargs)
                if reference_value is not None:
                    axes[idx_row, idx_col].plot(
                        reference_value[key_row], 0,
                        color='red',
                        alpha=1.0,
                        linewidth=2,
                        marker='X',
                        clip_on=False,
                        markersize=12)
                axes[idx_row, idx_col].get_yaxis().set_ticklabels([])
                axes[idx_row, idx_col].set(xlim=(min_samples, max_samples))
            else:
                if (idx_row > idx_col) or draw_upper_triagonal:
                    axes[idx_row, idx_col].plot(samples[key_col],
                                                samples[key_row],
                                                linestyle='',
                                                marker='o',
                                                alpha=0.6,
                                                clip_on=False,
                                                markersize=dot_size,
                                                markeredgecolor=edgecolor,
                                                **kwargs)
                    if reference_value is not None:
                        axes[idx_row, idx_col].plot(
                            [samples[key_col].min(), samples[key_col].max()],
                            [reference_value[key_row], reference_value[key_row]],
                            color='red', alpha=0.8, linewidth=2)
                        axes[idx_row, idx_col].plot(
                            [reference_value[key_col], reference_value[key_col]],
                            [samples[key_row].min(), samples[key_row].max()],
                            color='red', alpha=0.8, linewidth=2)

                    axes[idx_row, idx_col].axis([samples[key_col].min(),
                                                samples[key_col].max(),
                                                samples[key_row].min(),
                                                samples[key_row].max()])
                else:
                    if idx_row < idx_col:
                        axes[idx_row, idx_col].axis('off')

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

    cmap = plt.cm.get_cmap("Blues")
    for ix in range(n_plots):
        for jy in range(n_plots):
            if ix == jy:
                axes[jy, ix].scatter(x_evidence[:, ix], y_evidence, edgecolors='black', alpha=0.6)
                axes[jy, ix].get_yaxis().set_ticklabels([])
                axes[jy, ix].yaxis.tick_right()
                axes[jy, ix].set_ylabel('Discrepancy')
                axes[jy, ix].yaxis.set_label_position("right")

                if true_params is not None:
                    axes[jy, ix].plot([true_params[parameter_names[ix]],
                                      true_params[parameter_names[ix]]],
                                      [min(y_evidence), max(y_evidence)],
                                      color='red', alpha=1.0, linewidth=1)
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
                axes[jy, ix].scatter(x_evidence[:, ix],
                                     x_evidence[:, jy],
                                     color="red",
                                     alpha=0.7,
                                     s=5)

                if true_params is not None:
                    axes[jy, ix].plot([true_params[parameter_names[ix]],
                                      true_params[parameter_names[ix]]],
                                      [bounds[jy][0], bounds[jy][1]],
                                      color='red', alpha=1.0, linewidth=1)

                    axes[jy, ix].plot([bounds[ix][0], bounds[ix][1]],
                                      [true_params[parameter_names[jy]],
                                      true_params[parameter_names[jy]]],
                                      color='red', alpha=1.0, linewidth=1)

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


def plot_predicted_summaries(model=None,
                             summary_names=None,
                             n_samples=100,
                             seed=None,
                             bins=20,
                             axes=None,
                             add_observed=True,
                             draw_upper_triagonal=False,
                             **kwargs):
    """Pairplots of 1D summary statistics calculated from prior predictive distribution.

    Parameters
    ----------
    model: elfi.Model
        Model which is explored.
    summary_names: list of strings
        Summary statistics which are pairplotted.
    n_samples: int, optional
        How many samples are drawn from the model.
    bins : int, optional
        Number of bins in histograms.
    axes : one or an iterable of plt.Axes, optional
    add_observed: boolean, optional
        Add observed summary points in pairplots
    draw_upper_triagonal: boolean, optional
        Boolean indicating whether to draw symmetric upper triagonal part.


    """
    dot_size = kwargs.pop('s', 8)
    samples = model.generate(batch_size=n_samples, outputs=summary_names, seed=seed)
    reference_value = model.generate(with_values=model.observed, outputs=summary_names)
    reference_value = reference_value if add_observed else None
    plot_pairs(samples,
               selector=None,
               bins=bins,
               axes=axes,
               reference_value=reference_value,
               s=dot_size,
               draw_upper_triagonal=draw_upper_triagonal)


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
        self.finished = False

    def update_progressbar(self, iteration, total):
        """Print updated progress bar in console.

        Parameters
        ----------
        iteration : int
            Integer indicating completed iterations
        total : int
            Integer indicating total number of iterations

        """
        if iteration >= total:
            percent = ("{0:." + str(self.decimals) + "f}").\
                format(100.0)
            bar = self.fill * self.length
            if not self.finished:
                print('%s [%s] %s%% %s' % (self.prefix, bar, percent, self.suffix))
                self.finished = True
        elif total - self.scaling > 0:
            percent = ("{0:." + str(self.decimals) + "f}").\
                format(100 * ((iteration - self.scaling) / float(total - self.scaling)))
            filled_length = int(self.length * (iteration - self.scaling) // (total - self.scaling))
            bar = self.fill * filled_length + '-' * (self.length - filled_length)
            print('%s [%s] %s%% %s' % (self.prefix, bar, percent, self.suffix), end='\r')

    def reinit_progressbar(self, scaling=0, reinit_msg=""):
        """Reinitialize new round of progress bar.

        Parameters
        ----------
        scaling : int, optional
            Integer used to scale current and total iterations of the progress bar
        reinit_msg : str, optional
            Message printed before restarting an empty progess bar on a new line

        """
        self.scaling = scaling
        self.finished = False
        print(reinit_msg)
