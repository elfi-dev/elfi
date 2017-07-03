Good to know
============

Here we describe some important concepts related to ELFI. These will help in understanding
how to implement custom operations (such as simulators or summaries) and can potentially
save the user from some pitfalls.


ELFI model
----------

In ELFI, the priors, simulators, summaries, distances, etc. are called operations. ELFI
provides a convenient syntax of combining these operations into a network that is called
an `ElfiModel`_, where each node represents an operation. Basically, the `ElfiModel`_ is a
description of how different quantities needed in the inference are to be generated. The
structure of the network is a `directed acyclic graph (DAG)`_.

.. _`directed acyclic graph (DAG)`: https://en.wikipedia.org/wiki/Directed_acyclic_graph

.. _`ElfiModel`: api.html#elfi.ElfiModel


Operations
----------

Operations are functions (or more generally Python callables) in the nodes of the
ELFI model. Those nodes that deal directly with data, e.g. priors, simulators,
summaries and distances should return a numpy array of length ``batch_size`` that contains
their output.

If your operation does not produce data wrapped to numpy arrays, you can use the
`elfi.tools.vectorize`_ tool to achieve that. Note that sometimes it is required to specify
which arguments to the vectorized function will be constants and at other times also
specify the datatype (when automatic numpy array conversion does not produce desired
result). It is always good to check that the output is sane using the ``node.generate``
method.

.. _`elfi.tools.vectorize`: api.html#elfi.tools.vectorize

Reusing data
------------

The `OutputPool`_ object can be used to store the outputs of any node in the graph. Note
however that changing a node in the model will change the outputs of it's child nodes. In
Rejection sampling you can alter the child nodes of the nodes in the `OutputPool`_ and
safely reuse the `OutputPool`_ with the modified model. This is especially handy when
saving the simulations and trying out different summaries. BOLFI allows you to use the
stored data as initialization data.

However passing a modified model with the `OutputPool`_ of the original model will produce
biased results in other algorithms besides Rejection sampling. This is because more
advanced algorithms learn from previous results. If the results change in some way, so
will also the following parameter values and thus also their simulations and other nodes
that depend on them. The Rejection sampling does not suffer from this because it always
samples new parameter values directly from the priors, and therefore modified distance
outputs have no effect to the parameter values of any later simulations.

.. _`OutputPool`: api.html#elfi.OutputPool