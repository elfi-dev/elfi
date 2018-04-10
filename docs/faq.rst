Frequently Asked Questions
==========================

Below are answers to some common questions asked about ELFI.

*Q: My uniform prior* ``elfi.Prior('uniform', 1, 2)`` *does not seem to be right as it
produces outputs from the interval (1, 3).*

**A**: The distributions defined by strings are those from ``scipy.stats`` and follow
their definitions. There the uniform distribution uses the location/scale definition, so
the first argument defines the starting point of the interval and the second its length.

.. _vectorization:

*Q: What is vectorization in ELFI?*

**A**: Looping is relatively inefficient in Python, and so whenever possible, you should *vectorize*
your operations_. This means that repetitive computations are performed on a batch of data using
precompiled libraries (typically NumPy_), which effectively runs the loops in faster, compiled C-code.
ELFI supports vectorized operations, and due to the potentially huge saving in CPU-time it is
recommended to vectorize all user-code whenever possible.

.. _operations: good-to-know.html#operations
.. _NumPy: http://www.numpy.org/

For example, imagine you have a simulator that depends on a scalar parameter and produces a vector of 5
values. When this is used in ELFI with ``batch_size`` set to 1000, ELFI draws 1000 values from the
parameter's prior distribution and gives this *vector* to the simulator. Ideally, the simulator should
efficiently process all 1000 parameter cases in one go and output an array of shape (1000, 5). When using
vectorized operations in ELFI, the length (i.e. the first dimension) of all output arrays should equal
``batch_size``. Note that because of this the evaluation of summary statistics, distances etc. should
bypass the first dimension (e.g. with NumPy functions using ``axis=1`` in this case).

See ``elfi.examples`` for tips on how to vectorize simulators and work with ELFI. In case you are
unable to vectorize your simulator, you can use `elfi.tools.vectorize`_ to mimic
vectorized behaviour, though without the performance benefits. Finally, for simplicity vectorization
is not assumed (``batch_size=1`` by default).

.. _`elfi.tools.vectorize`: api.html#elfi.tools.vectorize

