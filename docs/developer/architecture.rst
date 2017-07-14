ELFI architecture
=================

Here we explain the internal representation of the ELFI model. This
representation contains everything that is needed to generate data, but is separate from
e.g. the inference methods or the data storages. This information is aimed for developers
and is not essential for using ELFI. We assume the reader is quite familiar with Python
and has perhaps already read some of ELFI's source code.

The low level representation of the ELFI model is a ``networkx.DiGraph`` with node names
as the nodes. The representation of the node is stored to the corresponding attribute
dictionary of the ``networkx.DiGraph``. We call this attribute dictionary the node *state*
dictionary. The ``networkx.DiGraph`` representation can be found from
``ElfiModel.source_net``. Before the ELFI model can be ran, it needs to be compiled and
loaded with data (e.g. observed data, precomputed data, batch index, batch size etc). The
compilation and loading of data is the responsibility of the ``Client`` implementation and
makes it possible in essence to translate ``ElfiModel`` to any kind of computational
backend. Finally the class ``Executor`` is responsible for running the compiled and loaded
model and producing the outputs of the nodes.

A user typically creates this low level representation by working with subclasses of
``NodeReference``. These are easy to use UI classes of ELFI such as the ``elfi.Simulator`` or
``elfi.Prior``. Under the hood they create proper node state dictionaries stored into the
``source_net``. The callables such as simulators or summaries that the user provides to
these classes are called operations.


The model graph representation
------------------------------

The ``source_net`` is a directed acyclic graph (DAG) and holds the state dictionaries of the
nodes and the edges between the nodes. An edge represents a dependency. For example and
edge from a prior node to the simulator node represents that the simulator requires a
value from the prior to be able to run. The edge name corresponds to a parameter name for
the operation, with integer names interpreted as positional parameters.

In the standard compilation process, the ``source_net`` is augmented with additional nodes
such as batch_size or random_state, that are then added as dependencies for those
operations that require them. In addition the state dicts will be turned into either a
runnable operation or a precomputed value.

The execution order of the nodes in the compiled graph follows the topological ordering of
the DAG (dependency order) and is guaranteed to be the same every time. Note that because
the default behaviour is that nodes share a random state, changing a node that uses a
shared random state will affect the result of any later node in the ordering using the
same random state, even if they would be independent based on the graph topology.


State dictionary
----------------

The state of a node is a Python dictionary. It describes the type of the node and any
other relevant state information, such as the user provided callable operation (e.g.
simulator or summary statistic) and any additional parameters the operation needs to be
provided in the compilation.

The following are reserved keywords of the state dict that serve as instructions for the
ELFI compiler. They begin with an underscore. Currently these are:

_operation : callable
    Operation of the node producing the output. Can not be used if _output is present.
_output : variable
    Constant output of the node. Can not be used if _operation is present.
_class : class
    The subclass of ``NodeReference`` that created the state.
_stochastic : bool, optional
    Indicates that the node is stochastic. ELFI will provide a random_state argument
    for such nodes, which contains a RandomState object for drawing random quantities.
    This node will appear in the computation graph. Using ELFI provided random states
    makes it possible to have repeatable experiments in ELFI.
_observable : bool, optional
    Indicates that there is observed data for this node or that it can be derived from the
    observed data. ELFI will create a corresponding observed node into the compiled graph.
    These nodes are dependencies of discrepancy nodes.
_uses_batch_size : bool, optional
    Indicates that the node operation requires ``batch_size`` as input. A corresponding edge
    from batch_size node to this node will be added to the compiled graph.
_uses_meta : bool, optional
    Indicates that the node operation requires meta information dictionary about the
    execution. This includes, model name, batch index and submission index.
    Useful for e.g. creating informative and unique file names. If the operation is
    vectorized with ``elfi.tools.vectorize``, then also ``index_in_batch`` will be added to
    the meta information dictionary.
_uses_observed : bool, optional
    Indicates that the node requires the observed data of its parents in the source_net as
    input. ELFI will gather the observed values of its parents to a tuple and link them to
    the node as a named argument observed.
_parameter : bool, optional
    Indicates that the node is a parameter node


The compilation and data loading phases
---------------------------------------

The compilation of the computation graph is separated from the loading of the data for
making it possible to reuse the compiled model. The subclasses of the ``Loader`` class
take responsibility of injecting data to the nodes of the compiled model. Examples of
injected data are precomputed values from the ``OutputPool``, the current ``random_state`` and
so forth.
