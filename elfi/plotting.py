import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_sample(samples, nodes=None, n=-1, displays=None, **options):
    axes = _prepare_axes(options)

    nodes = nodes or sorted(samples.keys())[:2]
    if isinstance(nodes, str): nodes = [nodes]

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
