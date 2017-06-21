Good to know
============

Here we describe some important concepts related to ELFI. These will help in understanding
how to implement custom operations (such as simulators or summaries) and can potentially
save the user from some pitfalls.

Generative model
----------------

By a generative model we mean any model that can generate some data. In ELFI the
generative model is described with a directed acyclic graph (DAG) and the representation
is stored in the `ElfiModel` instance. In LFI it typically includes everything from prior
distributions up to the summaries or distances.

Operations
----------

Operations are functions (or more generally Python callables) in the nodes of the
generative model. Those nodes that deal directly with data, e.g. priors, simulators,
summaries and distances should return a numpy array of length `batch_size` that contains
their output.

If your operation does not produce data wrapped to numpy arrays, you can use the
`elfi.tools.vectorize` tool to achieve that. Note that sometimes it is required to specify
which arguments to the vectorized function will be constants and at other times also
specify the datatype (when automatic numpy array conversion does not produce desired
result). It is always good to check that the output is sane using the `node.generate`
method.
