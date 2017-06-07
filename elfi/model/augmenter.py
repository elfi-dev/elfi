import functools
from functools import partial, reduce
from operator import mul

from toolz.functoolz import compose

from elfi.model.elfi_model import NodeReference, Operation
from elfi.utils import args_to_tuple


def add_pdf_nodes(model, joint=True, nodes=None):
    """Adds pdf nodes for distribution nodes to the model and returns the node names.

    By default this gives the pdfs of the generated model parameters.

    Parameters
    ----------
    model : elfi.ElfiModel
    joint : bool, optional
        If True (default) return a the joint pdf of the priors
    nodes : list, optional
        List of distribution node names. Default is `model.parameters`.

    Returns
    -------
    pdfs : list
        List of node names. Either only the joint pdf node name or the separate pdf node
        names depending on the `joint` argument.

    """
    nodes = nodes or model.parameters

    pdfs = []
    for n in nodes:
        node = model[n]
        pdfs.append(Operation(node.distribution.pdf, *([node] + node.parents),
                              model=model, name='_{}_pdf*'.format(n)))

    if joint:
        return [add_reduce_node(model, pdfs, mul, '_joint_pdf*')]
    else:
        return [pdf.name for pdf in pdfs]


def add_reduce_node(model, nodes, reduce_operation, name):
    """Reduce the output from a collection of nodes

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
    op = Operation(compose(partial(reduce, reduce_operation), args_to_tuple), *nodes,
                   model=model, name=name)
    return op.name