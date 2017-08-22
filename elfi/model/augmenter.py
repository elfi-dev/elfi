"""This module contains auxiliary functions for augmenting the ELFI graph."""

from functools import partial, reduce
from operator import add, mul

from toolz.functoolz import compose

from elfi.model.elfi_model import NodeReference, Operation
from elfi.utils import args_to_tuple


def add_pdf_gradient_nodes(model, log=False, nodes=None):
    """Add gradient nodes for distribution nodes to the model.

    Returns the node names.

    By default this gives the pdfs of the generated model parameters.

    Parameters
    ----------
    model : elfi.ElfiModel
    log : bool, optional
        Use gradient of logpdf, default False.
    nodes : list, optional
        List of distribution node names. Default is `model.parameters`.

    Returns
    -------
    gradients : list
        List of gradient node names.

    """
    nodes = nodes or model.parameter_names
    gradattr = 'gradient_pdf' if log is False else 'gradient_logpdf'

    grad_nodes = _add_distribution_nodes(model, nodes, gradattr)

    return [g.name for g in grad_nodes]


# TODO: check that there are no latent variables. See model.utils.ModelPrior
def add_pdf_nodes(model, joint=True, log=False, nodes=None):
    """Add pdf nodes for distribution nodes to the model.

    Returns the node names.

    By default this gives the pdfs of the generated model parameters.

    Parameters
    ----------
    model : elfi.ElfiModel
    joint : bool, optional
        If True (default) return a the joint pdf of the priors
    log : bool, optional
        Use logpdf, default False.
    nodes : list, optional
        List of distribution node names. Default is `model.parameters`.

    Returns
    -------
    pdfs : list
        List of node names. Either only the joint pdf node name or the separate pdf node
        names depending on the `joint` argument.

    """
    nodes = nodes or model.parameter_names
    pdfattr = 'pdf' if log is False else 'logpdf'

    pdfs = _add_distribution_nodes(model, nodes, pdfattr)

    if joint:
        if log:
            return [add_reduce_node(model, pdfs, add, '_joint_{}*'.format(pdfattr))]
        else:
            return [add_reduce_node(model, pdfs, mul, '_joint_{}*'.format(pdfattr))]
    else:
        return [pdf.name for pdf in pdfs]


def _add_distribution_nodes(model, nodes, attr):
    distribution_nodes = []
    for n in nodes:
        node = model[n]
        op = getattr(node.distribution, attr)
        distribution_nodes.append(
            Operation(op, *([node] + node.parents), model=model, name='_{}_{}'.format(n, attr)))
    return distribution_nodes


def add_reduce_node(model, nodes, reduce_operation, name):
    """Reduce the output from a collection of nodes.

    Parameters
    ----------
    model : elfi.ElfiModel
    nodes : list
        Either a list of node names or a list of node reference objects
    reduce_operation : callable
    name : str
        Name for the reduce node

    Returns
    -------
    name : str
        name of the new node

    """
    name = '_reduce*' if name is None else name
    nodes = [n if isinstance(n, NodeReference) else model[n] for n in nodes]
    op = Operation(
        compose(partial(reduce, reduce_operation), args_to_tuple), *nodes, model=model, name=name)
    return op.name
